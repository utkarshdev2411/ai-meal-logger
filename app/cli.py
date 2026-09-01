"""Interactive terminal chat loop for the text-path agent (FR-4.6).

`python -m app.cli` opens one DB session per turn, ensures a demo user exists,
runs the graph, and prints the reply. Thread state persists across turns
within this process via an in-memory checkpointer (`MemorySaver`) built once
at startup and threaded through every turn's fresh graph build.

After each reply, background memory extraction (FR-5.5) is kicked off via
`asyncio.create_task` — never awaited before the next prompt, so it adds 0ms
to the turn. It opens its OWN db session rather than reusing the turn's: the
turn's `async with async_session_factory()` block has already exited (and may
close/recycle its connection) by the time the background task actually runs.
Pending tasks are tracked and drained with `asyncio.gather` on clean exit so a
still-in-flight extraction isn't silently dropped.
"""

from __future__ import annotations

import argparse
import asyncio

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from app.agent.graph import build_graph
from app.db import repo
from app.db.client import async_session_factory, create_all
from app.memory.extractor import extract_and_write

DEMO_EXTERNAL_ID = "demo"
THREAD_ID = "cli"


async def run_turn(checkpointer: MemorySaver, user_id: str, text: str) -> str:
    async with async_session_factory() as session:
        graph = build_graph(session, user_id, checkpointer)
        config = {"configurable": {"thread_id": THREAD_ID}}
        # last_meal_id intentionally omitted here: it's a checkpointed channel
        # written by tool Commands, and re-sending it each turn would reset it.
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=text)], "user_id": user_id},
            config=config,
        )
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                return msg.content if isinstance(msg.content, str) else str(msg.content)
        return "(no reply)"


async def _run_extraction(user_id: str, text: str, reply: str) -> None:
    async with async_session_factory() as session:
        await extract_and_write(session, user_id, text, reply)


async def main() -> None:
    parser = argparse.ArgumentParser(description="CalorAI text-path CLI")
    parser.add_argument("--image", help="path to an image")
    args = parser.parse_args()
    if args.image:
        print(f"image support not wired yet ({args.image} ignored) — text only this phase.")

    await create_all()
    checkpointer = MemorySaver()
    async with async_session_factory() as session:
        user = await repo.get_or_create_user(session, DEMO_EXTERNAL_ID)
    user_id = user.id

    background_tasks: set[asyncio.Task] = set()
    print("CalorAI — type a message (Ctrl-D to quit)")
    while True:
        try:
            text = input("you> ").strip()
        except EOFError:
            print()
            break
        if not text:
            continue
        reply = await run_turn(checkpointer, user_id, text)
        print(f"bot> {reply}")

        task = asyncio.create_task(_run_extraction(user_id, text, reply))
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)

    if background_tasks:
        await asyncio.gather(*background_tasks, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())
