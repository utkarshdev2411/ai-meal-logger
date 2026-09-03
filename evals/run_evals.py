from __future__ import annotations

import asyncio
import itertools
import os
import re
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCRATCH_DB = Path(tempfile.mktemp(suffix=".sqlite3"))
os.environ.setdefault("LLM_API_KEY", "evals-script-placeholder")
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{SCRATCH_DB}"

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

from app.agent.graph import build_graph  # noqa: E402
from app.db import repo  # noqa: E402
from app.db.client import async_session_factory, create_all, engine  # noqa: E402
from app.db.models import Base, Image  # noqa: E402
from app.mealops.logging_ops import log_meal as run_log_meal  # noqa: E402
from app.memory.store import write_memory  # noqa: E402
from app.nutrition.resolve import RawItem, resolve  # noqa: E402

CASES_PATH = Path(__file__).parent / "cases.yaml"
USER_EXTERNAL_ID = "eval-user"
TOLERANCE = 0.5

PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02"
    b"\x00\x00\x00\x90wS\xde\x00\x00\x00\nIDATx\x9cc\xf8\xcf\xc0\x00\x00\x03\x01"
    b"\x01\x00\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
)


class ScriptedLLM:
    def __init__(self, queue: list, sink: list[str]):
        self._queue = queue
        self._sink = sink

    def bind_tools(self, tools, **kwargs):
        return self

    async def ainvoke(self, messages, **kwargs):
        msg = self._queue.pop(0)
        for tc in getattr(msg, "tool_calls", None) or []:
            self._sink.append(tc["name"])
        return msg


def tool_call_msg(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def find_item_id(messages: list, item_name: str) -> str:
    pattern = re.compile(rf"{re.escape(item_name)} x[\d.]+\S* \(item_id=([\w-]+)")
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            match = pattern.search(msg.content)
            if match:
                return match.group(1)
    raise AssertionError(f"no item_id found for {item_name!r} in prior tool messages")


def resolve_placeholders(value: Any, messages: list) -> Any:
    if isinstance(value, str) and value.startswith("$item:"):
        return find_item_id(messages, value.split(":", 1)[1])
    if isinstance(value, dict):
        return {k: resolve_placeholders(v, messages) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_placeholders(v, messages) for v in value]
    return value


def build_queue(llm_steps: list[dict], messages_so_far: list, call_counter: itertools.count) -> list:
    queue = []
    for step in llm_steps:
        if "tool" in step:
            args = resolve_placeholders(step.get("args", {}), messages_so_far)
            call_id = f"call_{next(call_counter)}"
            queue.append(tool_call_msg(step["tool"], args, call_id))
        else:
            queue.append(AIMessage(content=step["reply"]))
    return queue


async def reset_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await create_all()


async def make_image(session, user_id: str, status: str, observation: dict | None) -> str:
    tmp = tempfile.mktemp(suffix=".png")
    Path(tmp).write_bytes(PNG_BYTES)
    image = Image(
        user_id=user_id,
        path=tmp,
        mime="image/png",
        width=1,
        height=1,
        bytes=len(PNG_BYTES),
        status=status,
        observation=observation,
        created_at=datetime.now(timezone.utc),
    )
    session.add(image)
    await session.commit()
    return image.id


async def apply_setup(session, user_id: str, setup: dict) -> dict[str, str]:
    for meal in setup.get("meals", []):
        when = datetime.now().astimezone() - timedelta(days=meal.get("days_ago", 0))
        await run_log_meal(
            session,
            user_id,
            meal["items"],
            meal_slot=meal["meal_slot"],
            when=when,
            source=meal.get("source", "text"),
        )
    for mem in setup.get("memories", []):
        await write_memory(
            session,
            user_id=user_id,
            kind=mem["kind"],
            key=mem["key"],
            value=mem["value"],
            confidence=mem.get("confidence", 1.0),
            source_message=mem.get("source_message"),
        )
    image_ids: dict[str, str] = {}
    for img in setup.get("images", []):
        image_ids[img["key"]] = await make_image(
            session, user_id, img.get("status", "ready"), img.get("observation")
        )
    return image_ids


async def compute_expected_totals(items_spec: list[dict]) -> tuple[float, float, float, float]:
    raw = [RawItem(name=i["name"], quantity=i.get("quantity", 1.0), unit=i.get("unit")) for i in items_spec]
    resolved = await resolve(raw)
    return (
        sum(r.kcal for r in resolved),
        sum(r.protein_g for r in resolved),
        sum(r.carbs_g for r in resolved),
        sum(r.fat_g for r in resolved),
    )


class CaseResult:
    def __init__(self, case_id: str):
        self.case_id = case_id
        self.failures: list[str] = []

    @property
    def passed(self) -> bool:
        return not self.failures


async def check_expect_db(result: CaseResult, session, user_id: str, expect_db: dict) -> None:
    if "meal_count" in expect_db:
        meals = await repo.recent_meals(session, user_id, since_date=date(2000, 1, 1))
        actual = len(meals)
        expected = expect_db["meal_count"]
        if actual != expected:
            result.failures.append(f"meal_count: expected {expected}, got {actual}")

    if "totals_items" in expect_db:
        exp_kcal, exp_protein, exp_carbs, exp_fat = await compute_expected_totals(expect_db["totals_items"])
        actual_totals = await repo.daily_totals(session, user_id, date.today())
        for label, expected_val, actual_val in (
            ("kcal", exp_kcal, actual_totals.kcal),
            ("protein_g", exp_protein, actual_totals.protein_g),
            ("carbs_g", exp_carbs, actual_totals.carbs_g),
            ("fat_g", exp_fat, actual_totals.fat_g),
        ):
            if abs(float(actual_val) - expected_val) > TOLERANCE:
                result.failures.append(
                    f"daily_totals.{label}: expected {expected_val:.1f}, got {float(actual_val):.1f}"
                )

    if "memories" in expect_db:
        actives = await repo.get_active_memories(session, user_id)
        for spec in expect_db["memories"]:
            match = next((m for m in actives if m.kind == spec["kind"] and m.key == spec["key"]), None)
            if match is None:
                result.failures.append(f"memory {spec['kind']}/{spec['key']}: no active row found")
                continue
            if spec.get("status", "active") != match.status:
                result.failures.append(
                    f"memory {spec['kind']}/{spec['key']}: expected status {spec.get('status')}, got {match.status}"
                )
            needle = spec.get("value_contains")
            if needle and needle.lower() not in str(match.value).lower():
                result.failures.append(
                    f"memory {spec['kind']}/{spec['key']}: value {match.value!r} does not contain {needle!r}"
                )

    if "last_meal_items" in expect_db:
        meal = await repo.last_active_meal(session, user_id)
        if meal is None:
            result.failures.append("last_meal_items: no active meal found")
        else:
            for spec in expect_db["last_meal_items"]:
                item = next((i for i in meal.items if i.name == spec["name"]), None)
                if item is None:
                    result.failures.append(f"last_meal_items: no item named {spec['name']!r}")
                elif abs(float(item.quantity) - spec["quantity"]) > 1e-6:
                    result.failures.append(
                        f"last_meal_items.{spec['name']}.quantity: expected {spec['quantity']}, got {float(item.quantity)}"
                    )

    if "total_item_count" in expect_db:
        meal = await repo.last_active_meal(session, user_id)
        actual = len(meal.items) if meal else 0
        expected = expect_db["total_item_count"]
        if actual != expected:
            result.failures.append(f"total_item_count: expected {expected}, got {actual}")


async def run_case(case: dict) -> CaseResult:
    result = CaseResult(case["id"])
    await reset_db()

    async with async_session_factory() as session:
        user = await repo.get_or_create_user(session, USER_EXTERNAL_ID)
        user_id = user.id
        image_ids = await apply_setup(session, user_id, case.get("setup", {}))

    checkpointer = MemorySaver()
    thread_id = f"case-{case['id']}"
    tool_calls_seen: list[str] = []
    messages_so_far: list = []
    call_counter = itertools.count()

    for turn in case["turns"]:
        queue = build_queue(turn["llm"], messages_so_far, call_counter)
        llm = ScriptedLLM(queue, tool_calls_seen)
        async with async_session_factory() as session:
            graph = build_graph(session, user_id, checkpointer, llm=llm)
            state_update: dict = {
                "messages": [HumanMessage(content=turn.get("user", ""))],
                "user_id": user_id,
            }
            if turn.get("image"):
                state_update["image_id"] = image_ids[turn["image"]]
            turn_result = await graph.ainvoke(
                state_update, config={"configurable": {"thread_id": thread_id}}
            )
        messages_so_far = turn_result["messages"]

    expect = case.get("expect", {})
    expected_tools = expect.get("tool_calls", [])
    if tool_calls_seen != expected_tools:
        result.failures.append(f"tool_calls: expected {expected_tools}, got {tool_calls_seen}")

    forbidden = set(expect.get("forbidden_tools", []))
    hit = forbidden & set(tool_calls_seen)
    if hit:
        result.failures.append(f"forbidden tool(s) called: {sorted(hit)}")

    async with async_session_factory() as session:
        await check_expect_db(result, session, user_id, case.get("expect_db", {}))

    return result


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)]
    def fmt_row(cells: list[str]) -> str:
        return " | ".join(c.ljust(w) for c, w in zip(cells, widths))
    lines = [fmt_row(headers), "-+-".join("-" * w for w in widths)]
    lines.extend(fmt_row(r) for r in rows)
    return "\n".join(lines)


async def main() -> None:
    cases = yaml.safe_load(CASES_PATH.read_text())
    await create_all()

    results: list[CaseResult] = []
    for case in cases:
        results.append(await run_case(case))

    SCRATCH_DB.unlink(missing_ok=True)

    rows = [[r.case_id, "PASS" if r.passed else "FAIL"] for r in results]
    print(render_table(["case", "status"], rows))
    print()

    failed = [r for r in results if not r.passed]
    if failed:
        print(f"{len(failed)}/{len(results)} case(s) FAILED\n")
        for r in failed:
            print(f"-- {r.case_id} --")
            for detail in r.failures:
                print(f"   {detail}")
        sys.exit(1)

    print(f"all {len(results)} case(s) passed")


if __name__ == "__main__":
    asyncio.run(main())
