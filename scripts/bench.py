from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCRATCH_DB = Path(tempfile.mktemp(suffix=".sqlite3"))
os.environ.setdefault("LLM_API_KEY", "bench-script-placeholder")
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{SCRATCH_DB}"

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

from app.agent.graph import build_graph  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.db import repo  # noqa: E402
from app.db.client import async_session_factory, create_all  # noqa: E402
from app.db.models import Image  # noqa: E402
from app.telemetry import TurnTimer, consume_cold, flush, new_turn_id  # noqa: E402
from app.vision.schema import VisionObservation  # noqa: E402

PHASES = ["prefetch", "vision", "llm_1", "tool", "llm_2", "ttft", "total"]

GOOD_OBS = VisionObservation.model_validate(
    {
        "items": [
            {"name": "dal", "portion_estimate": "1 bowl", "confidence": 0.9, "alternatives": []},
            {"name": "rice", "portion_estimate": "1 cup", "confidence": 0.85, "alternatives": []},
        ],
        "plate_context": "thali",
        "overall_confidence": 0.87,
        "unclear": [],
    }
)


class ScriptedLLM:
    def __init__(self, queue: list):
        self._queue = queue

    def bind_tools(self, tools, **kwargs):
        return self

    async def ainvoke(self, messages, **kwargs):
        return self._queue.pop(0)


def tool_call_msg(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


async def run_turn(
    user_id: str,
    checkpointer: MemorySaver,
    thread_id: str,
    path: str,
    text: str,
    queue: list | None,
    image_id: str | None = None,
    real: bool = False,
) -> tuple[list, str]:
    llm = None if real else ScriptedLLM(queue)
    settings = get_settings()
    timer = TurnTimer(
        turn_id=new_turn_id(),
        path=path,
        user_id=user_id,
        fast_path=settings.fast_path,
        cold=consume_cold(),
    )
    start = time.perf_counter()
    async with async_session_factory() as session:
        graph = build_graph(session, user_id, checkpointer, llm=llm, timer=timer)
        state_update: dict = {"messages": [HumanMessage(content=text)], "user_id": user_id}
        if image_id:
            state_update["image_id"] = image_id
        result = await graph.ainvoke(
            state_update, config={"configurable": {"thread_id": thread_id}}
        )
    total_ms = int((time.perf_counter() - start) * 1000)
    timer.record("total", total_ms)
    timer.record("ttft", total_ms)
    return result["messages"], timer.turn_id


async def make_image(session, user_id: str, status: str, observation: dict | None) -> str:
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02"
        b"\x00\x00\x00\x90wS\xde\x00\x00\x00\nIDATx\x9cc\xf8\xcf\xc0\x00\x00\x03\x01"
        b"\x01\x00\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    tmp = tempfile.mktemp(suffix=".png")
    Path(tmp).write_bytes(png_bytes)
    image_id = str(uuid.uuid4())
    session.add(
        Image(
            id=image_id,
            user_id=user_id,
            path=tmp,
            mime="image/png",
            width=1,
            height=1,
            bytes=len(png_bytes),
            status=status,
            observation=observation,
            created_at=datetime.now(timezone.utc),
        )
    )
    await session.commit()
    return image_id


def item_id_from_tool_message(messages: list, tool_name: str, item_name: str) -> str:
    tool_msg = next(m for m in messages if isinstance(m, ToolMessage) and m.name == tool_name)
    match = re.search(rf"{item_name} x[\d.]+\S* \(item_id=([\w-]+)", tool_msg.content)
    if not match:
        raise AssertionError(f"could not find {item_name} item_id in: {tool_msg.content}")
    return match.group(1)


async def bench_log_intent(user_id: str, iteration: int, turn_ids: list[str], real: bool = False) -> None:
    queue = None if real else [
        tool_call_msg(
            "log_meal",
            {
                "items": [
                    {"name": "paratha", "quantity": 2, "unit": "piece"},
                    {"name": "chai", "quantity": 1},
                ],
                "meal_slot": "breakfast",
            },
            "call_1",
        ),
        AIMessage(content="Logged 2 parathas and chai for breakfast."),
    ]
    _, turn_id = await run_turn(
        user_id,
        MemorySaver(),
        f"bench-log-{iteration}",
        "text",
        "had 2 parathas and chai for breakfast",
        queue,
        real=real,
    )
    turn_ids.append(turn_id)


async def bench_query_intent(user_id: str, iteration: int, turn_ids: list[str], real: bool = False) -> None:
    queue = None if real else [AIMessage(content="You're at about 520 kcal today.")]
    _, turn_id = await run_turn(
        user_id,
        MemorySaver(),
        f"bench-query-{iteration}",
        "text",
        "how am I doing on calories?",
        queue,
        real=real,
    )
    turn_ids.append(turn_id)


async def bench_correction_intent(user_id: str, iteration: int, turn_ids: list[str], real: bool = False) -> None:
    checkpointer = MemorySaver()
    thread_id = f"bench-correction-{iteration}"

    if real:
        await run_turn(
            user_id, checkpointer, thread_id, "text", "had 2 parathas for breakfast", None, real=True
        )
        _, turn_id = await run_turn(
            user_id, checkpointer, thread_id, "text", "actually that was 3 rotis not 2", None, real=True
        )
        turn_ids.append(turn_id)
        return

    setup_queue = [
        tool_call_msg(
            "log_meal",
            {"items": [{"name": "paratha", "quantity": 2, "unit": "piece"}], "meal_slot": "breakfast"},
            "call_setup",
        ),
        AIMessage(content="Logged."),
    ]
    messages, _ = await run_turn(
        user_id, checkpointer, thread_id, "text", "had 2 parathas for breakfast", setup_queue
    )
    item_id = item_id_from_tool_message(messages, "log_meal", "paratha")

    correction_queue = [
        tool_call_msg(
            "revise_meal",
            {"action": "set_item_qty", "meal_ref": "last", "item_id": item_id, "quantity": 3},
            "call_2",
        ),
        AIMessage(content="Updated to 3 parathas — corrected, not doubled."),
    ]
    _, turn_id = await run_turn(
        user_id, checkpointer, thread_id, "text", "actually that was 3 rotis not 2", correction_queue
    )
    turn_ids.append(turn_id)


async def bench_image_prewarmed(user_id: str, iteration: int, turn_ids: list[str], real: bool = False) -> None:
    async with async_session_factory() as session:
        image_id = await make_image(session, user_id, "ready", GOOD_OBS.model_dump())

    queue = None if real else [
        tool_call_msg(
            "log_meal",
            {
                "items": [{"name": "dal", "quantity": 0.5}, {"name": "rice", "quantity": 0.5}],
                "meal_slot": "dinner",
            },
            "call_1",
        ),
        AIMessage(content="Logged half a thali — dal and rice."),
    ]
    _, turn_id = await run_turn(
        user_id,
        MemorySaver(),
        f"bench-image-warm-{iteration}",
        "image",
        "half of this was my brother's",
        queue,
        image_id=image_id,
        real=real,
    )
    turn_ids.append(turn_id)


async def bench_image_cold(user_id: str, iteration: int, turn_ids: list[str], real: bool = False) -> None:
    async with async_session_factory() as session:
        image_id = await make_image(session, user_id, "pending", None)

    if real:
        _, turn_id = await run_turn(
            user_id,
            MemorySaver(),
            f"bench-image-cold-{iteration}",
            "image",
            "half of this was my brother's",
            None,
            image_id=image_id,
            real=True,
        )
        turn_ids.append(turn_id)
        return

    queue = [
        tool_call_msg(
            "log_meal",
            {
                "items": [{"name": "dal", "quantity": 0.5}, {"name": "rice", "quantity": 0.5}],
                "meal_slot": "dinner",
            },
            "call_1",
        ),
        AIMessage(content="Logged half a thali — dal and rice."),
    ]
    with patch(
        "app.vision.process.extract_vision", new=AsyncMock(return_value=GOOD_OBS)
    ):
        _, turn_id = await run_turn(
            user_id,
            MemorySaver(),
            f"bench-image-cold-{iteration}",
            "image",
            "half of this was my brother's",
            queue,
            image_id=image_id,
        )
    turn_ids.append(turn_id)


def percentile(values: list[int], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    if lo == hi:
        return float(ordered[lo])
    frac = k - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


async def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="CalorAI latency bench (scripted LLM by default)")
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--real", action="store_true", help="use the real configured LLM, not a scripted one")
    args = parser.parse_args()
    n = args.runs if args.runs is not None else (8 if args.real else settings.bench_default_runs)

    await create_all()
    async with async_session_factory() as session:
        user = await repo.get_or_create_user(session, "bench-user")
    user_id = user.id

    cases = [
        ("log intent (text)", bench_log_intent, "text"),
        ("query intent (text)", bench_query_intent, "text"),
        ("correction intent (text)", bench_correction_intent, "text"),
        ("photo + caption, prewarmed", bench_image_prewarmed, "image"),
        ("photo + caption, cold", bench_image_cold, "image"),
    ]

    case_turn_ids: dict[str, list[str]] = {name: [] for name, _, _ in cases}
    failures: list[str] = []

    for name, fn, _ in cases:
        for i in range(n):
            try:
                await fn(user_id, i, case_turn_ids[name], real=args.real)
            except Exception as exc:
                failures.append(f"{name} run {i}: {type(exc).__name__}: {exc}")

    await flush()

    all_turn_ids = [t for ids in case_turn_ids.values() for t in ids]
    async with async_session_factory() as session:
        samples = await repo.get_latency_samples(session, turn_ids=all_turn_ids)

    turn_id_to_case = {t: name for name, ids in case_turn_ids.items() for t in ids}
    turn_id_to_path = {t: path for name, _, path in cases for t in case_turn_ids[name]}

    if args.real:
        print(f"CalorAI latency bench — measured against the real configured LLM ({settings.text_model}).")
        print("Real network, real model latency, real key-pool round-robin. Honest numbers.")
    else:
        print("CalorAI latency bench — measured against a scripted/fake LLM (no network, no real")
        print("model latency). Real DB, prefetch, and vision-stub timing; NOT final honest numbers.")
        print("Re-run with --real once a real LLM_API_KEY is set to get those.")
    print()
    print(f"runs per case: {n}, db_backend={samples[0].db_backend if samples else 'sqlite'}")
    if failures:
        print(f"\n{len(failures)} run(s) failed and were skipped:")
        for f in failures:
            print(f"  - {f}")
    print()

    print("## Totals per case\n")
    total_rows = []
    for name, _, path in cases:
        warm = [s.ms for s in samples if turn_id_to_case.get(s.turn_id) == name and s.phase == "total" and not s.cold]
        ttft = [s.ms for s in samples if turn_id_to_case.get(s.turn_id) == name and s.phase == "ttft" and not s.cold]
        total_rows.append(
            [
                name,
                str(len(warm)),
                f"{percentile(ttft, 0.5):.0f}",
                f"{percentile(warm, 0.5):.0f}",
                f"{percentile(warm, 0.95):.0f}",
            ]
        )
    print(render_table(["case", "n (warm)", "TTFT p50 ms", "total p50 ms", "total p95 ms"], total_rows))
    print()

    print("## Phase breakdown by path\n")
    phase_rows = []
    for path in ["text", "image"]:
        for phase in PHASES:
            values = [
                s.ms
                for s in samples
                if turn_id_to_path.get(s.turn_id) == path and s.phase == phase and not s.cold
            ]
            if not values:
                continue
            phase_rows.append(
                [
                    path,
                    phase,
                    str(len(values)),
                    f"{percentile(values, 0.5):.0f}",
                    f"{percentile(values, 0.95):.0f}",
                ]
            )
    print(render_table(["path", "phase", "n", "p50 ms", "p95 ms"], phase_rows))
    print()

    cold_rows = []
    for s in samples:
        if s.cold:
            cold_rows.append([turn_id_to_case.get(s.turn_id, "?"), s.phase, str(s.ms)])
    if cold_rows:
        print("## Cold-start sample (first turn after process start, excluded above)\n")
        print(render_table(["case", "phase", "ms"], cold_rows))
        print()

    SCRATCH_DB.unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(main())
