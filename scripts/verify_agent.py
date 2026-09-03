"""Smoke test for the agent core."""


from __future__ import annotations

import asyncio
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCRATCH_DB = Path(__file__).resolve().parent.parent / "scratch_verify_agent.db"
os.environ.setdefault("LLM_API_KEY", "verify-script-placeholder")
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{SCRATCH_DB}"

if SCRATCH_DB.exists():
    SCRATCH_DB.unlink()

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

from app.agent.graph import build_graph  # noqa: E402
from app.db import repo  # noqa: E402
from app.db.client import async_session_factory, create_all  # noqa: E402


class ScriptedLLM:
    """Duck-typed stand-in for `ChatOpenAI` — `bind_tools` is a no-op, `ainvoke`
    pops the next canned response. The queue is a shared list so the script
    can append later turns' scripted calls after inspecting earlier results
    (needed for the item_id handoff, which isn't known until runtime)."""

    def __init__(self, queue: list):
        self._queue = queue

    def bind_tools(self, tools, **kwargs):
        return self

    async def ainvoke(self, messages, **kwargs):
        return self._queue.pop(0)


def tool_call_msg(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


async def run_turn(graph, thread_id: str, text: str) -> tuple[str, list[str]]:
    config = {"configurable": {"thread_id": thread_id}}
    result = await graph.ainvoke({"messages": [HumanMessage(content=text)], "user_id": "verify-user"}, config=config)
    tool_names = [m.name for m in result["messages"] if isinstance(m, ToolMessage) and getattr(m, "name", None)]
    final = next((m.content for m in reversed(result["messages"]) if isinstance(m, AIMessage) and m.content), "")
    return final, tool_names, result["messages"]


async def main() -> None:
    await create_all()
    async with async_session_factory() as session:
        user = await repo.get_or_create_user(session, "verify-agent")
    user_id = user.id

    checkpointer = MemorySaver()
    queue: list = []
    fake = ScriptedLLM(queue)

    async def new_graph():
        async with async_session_factory() as session:
            return session, build_graph(session, user_id, checkpointer, llm=fake)

    print("=== turn 1: log a meal ===")
    queue.append(
        tool_call_msg(
            "log_meal",
            {"items": [{"name": "paratha", "quantity": 2, "unit": "piece"}, {"name": "chai", "quantity": 1}], "meal_slot": "breakfast"},
            "call_1",
        )
    )
    queue.append(AIMessage(content="Logged 2 parathas and chai for breakfast."))
    async with async_session_factory() as session:
        graph = build_graph(session, user_id, checkpointer, llm=fake)
        reply, tools_called, messages = await run_turn(graph, "verify", "had 2 parathas and chai for breakfast")
    print("bot>", reply)
    print("tools called:", tools_called)

    tool_msg = next(m for m in messages if isinstance(m, ToolMessage) and m.name == "log_meal")
    match = re.search(r"paratha x[\d.]+\S* \(item_id=([\w-]+)", tool_msg.content)
    assert match, f"could not find paratha item_id in: {tool_msg.content}"
    paratha_item_id = match.group(1)
    print("captured item_id for paratha:", paratha_item_id)

    print("\n=== turn 2: correction (no double-count) ===")
    queue.append(tool_call_msg("revise_meal", {"action": "set_item_qty", "meal_ref": "last", "item_id": paratha_item_id, "quantity": 3}, "call_2"))
    queue.append(AIMessage(content="Updated to 3 parathas — corrected, not doubled."))
    async with async_session_factory() as session:
        graph = build_graph(session, user_id, checkpointer, llm=fake)
        reply, tools_called, messages = await run_turn(graph, "verify", "actually that was 3 rotis not 2")
    print("bot>", reply)
    print("tools called:", tools_called)

    print("\n=== turn 3: query totals ===")
    queue.append(tool_call_msg("get_daily_totals", {}, "call_3"))
    queue.append(AIMessage(content="You're at about 520 kcal today."))
    async with async_session_factory() as session:
        graph = build_graph(session, user_id, checkpointer, llm=fake)
        reply, tools_called, messages = await run_turn(graph, "verify", "how am I doing on calories?")
    print("bot>", reply)
    print("tools called:", tools_called)

    async with async_session_factory() as session:
        from datetime import date

        from app.mealops.logging_ops import get_totals

        totals = await get_totals(session, user_id, day=date.today())
        assert len(totals.meals) == 1, f"expected exactly one meal (no double-count), got {len(totals.meals)}"
        item = totals.meals[0].items[0]
        assert item.quantity == 3, f"expected quantity 3 after correction, got {item.quantity}"
        print(f"\nDB check OK: {len(totals.meals)} meal, item quantity={item.quantity}, daily kcal={totals.totals.kcal:.0f}")

    SCRATCH_DB.unlink(missing_ok=True)
    print("\nverify_agent.py: all checks passed")


if __name__ == "__main__":
    asyncio.run(main())
