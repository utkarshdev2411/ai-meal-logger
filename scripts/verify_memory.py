"""Verification script for the memory layer."""


from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCRATCH_DB = Path(__file__).resolve().parent.parent / "scratch_verify_memory.db"
os.environ.setdefault("LLM_API_KEY", "verify-script-placeholder")
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{SCRATCH_DB}"

if SCRATCH_DB.exists():
    SCRATCH_DB.unlink()

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402

from app.agent.graph import build_graph  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.db import repo  # noqa: E402
from app.db.client import create_all  # noqa: E402
from app.db.models import Base  # noqa: E402
from app.memory.extractor import extract_and_write  # noqa: E402
from app.memory.store import render_memory_block, retrieve_memories, write_memory  # noqa: E402

results: list[tuple[str, bool]] = []


def check(name: str, condition: bool) -> None:
    results.append((name, condition))
    print(f"{'PASS' if condition else 'FAIL'} — {name}")


class ScriptedLLM:
    """Duck-typed `ChatOpenAI` stand-in — same pattern as verify_agent.py."""

    def __init__(self, queue: list):
        self._queue = queue
        self.last_prompt: list = []

    def bind_tools(self, tools, **kwargs):
        return self

    async def ainvoke(self, messages, **kwargs):
        self.last_prompt = messages
        return self._queue.pop(0)


async def main() -> None:
    await create_all()
    engine = create_async_engine(os.environ["DATABASE_URL"])
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with session_factory() as session:
        user = await repo.get_or_create_user(session, "verify-memory")
        user_id = user.id

        # --- 1. write_memory + retrieve_memories round-trip ------------------
        await write_memory(
            session, user_id, kind="diet", key="vegetarian",
            value={"text": "vegetarian"}, confidence=1.0, source_message="i'm vegetarian btw",
        )
        retrieved = await retrieve_memories(session, user_id)
        check(
            "write_memory + retrieve_memories round-trip",
            any(m.kind == "diet" and m.key == "vegetarian" for m in retrieved),
        )

        # --- 3. conflicting facts supersede via write_memory ------------------
        await write_memory(
            session, user_id, kind="goal", key="protein_target",
            value={"metric": "protein_g", "target": 120, "period": "day"},
            confidence=1.0, source_message="target 120g protein",
        )
        await write_memory(
            session, user_id, kind="goal", key="protein_target",
            value={"metric": "protein_g", "target": 140, "period": "day"},
            confidence=1.0, source_message="actually make it 140g protein",
        )
        active_goals = [
            m for m in await repo.get_active_memories(session, user_id)
            if m.kind == "goal" and m.key == "protein_target"
        ]
        check(
            "conflicting facts supersede: exactly one active row via write_memory",
            len(active_goals) == 1 and active_goals[0].value["target"] == 140,
        )

        # --- 4. many facts still fit the token budget -------------------------
        settings = get_settings()
        for i in range(60):
            await write_memory(
                session, user_id, kind="preference", key=f"pref_{i}",
                value={"text": f"synthetic preference number {i} about portion sizing habits"},
                confidence=0.8, source_message="synthetic",
            )
        many = await retrieve_memories(session, user_id)
        block = render_memory_block(many)
        block_tokens = len(block) // 4
        check(
            f"50+ facts still respect MEMORY_TOKEN_BUDGET ({block_tokens} <= {settings.memory_token_budget})",
            block_tokens <= settings.memory_token_budget,
        )
        check("retrieve_memories caps at MEMORY_TOP_K", len(many) <= settings.memory_top_k)

    await engine.dispose()

    # --- 2. cross-session persistence: fresh engine/session against same file
    engine2 = create_async_engine(os.environ["DATABASE_URL"])
    session_factory2 = async_sessionmaker(engine2, expire_on_commit=False)
    async with session_factory2() as session2:
        fresh = await retrieve_memories(session2, user_id, top_k=20)
        check(
            "cross-session persistence: vegetarian fact visible from a fresh connection",
            any(m.kind == "diet" and m.key == "vegetarian" for m in fresh),
        )
    await engine2.dispose()

    # --- 5. a stored fact reaches the real graph's system prompt -------------
    engine3 = create_async_engine(os.environ["DATABASE_URL"])
    session_factory3 = async_sessionmaker(engine3, expire_on_commit=False)
    async with session_factory3() as session3:
        checkpointer = MemorySaver()
        fake = ScriptedLLM([AIMessage(content="Sounds good.")])
        graph = build_graph(session3, user_id, checkpointer, llm=fake)
        await graph.ainvoke(
            {"messages": [HumanMessage(content="what should I eat")], "user_id": user_id},
            config={"configurable": {"thread_id": "verify-memory-graph"}},
        )
        system_msgs = [m for m in fake.last_prompt if isinstance(m, SystemMessage)]
        prompt_text = system_msgs[0].content if system_msgs else ""
        check(
            "graph's rendered system prompt contains the vegetarian fact",
            "vegetarian" in prompt_text,
        )
    await engine3.dispose()

    # --- 6. extract_and_write never raises, and writes a genuine proposal ----
    engine4 = create_async_engine(os.environ["DATABASE_URL"])
    session_factory4 = async_sessionmaker(engine4, expire_on_commit=False)
    async with session_factory4() as session4:
        with patch("httpx.AsyncClient.post", new=AsyncMock(side_effect=OSError("network down"))):
            try:
                await extract_and_write(session4, user_id, "had some food", "logged it")
                extractor_never_raised = True
            except Exception:
                extractor_never_raised = False
        check("extract_and_write never raises on a network failure", extractor_never_raised)

        class _FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"facts": [{"kind": "goal", "key": "calorie_target", '
                                    '"value": {"metric": "kcal", "target": 2000, "period": "day"}, '
                                    '"confidence": 0.9}]}'
                                )
                            }
                        }
                    ]
                }

        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=_FakeResponse())):
            await extract_and_write(session4, user_id, "aim for 2000 kcal a day", "noted")
        written = [
            m for m in await repo.get_active_memories(session4, user_id)
            if m.kind == "goal" and m.key == "calorie_target"
        ]
        check("extract_and_write persists a genuinely proposed fact via write_memory", len(written) == 1)
    await engine4.dispose()

    SCRATCH_DB.unlink(missing_ok=True)

    failed = [name for name, ok in results if not ok]
    print()
    if failed:
        print(f"{len(failed)} check(s) FAILED: {failed}")
        sys.exit(1)
    print(f"All {len(results)} checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
