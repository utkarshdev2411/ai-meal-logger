"""Interactive terminal chat loop for the text-path agent (FR-4.6).

`python -m app.cli` opens one DB session per turn, ensures a demo user exists,
runs the graph, and prints the reply. Thread state persists across turns
within this process via an in-memory checkpointer (`MemorySaver`) built once
at startup and threaded through every turn's fresh graph build.
"""

from __future__ import annotations

import argparse
import asyncio

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from app.agent.graph import build_graph
from app.db import repo
from app.db.client import async_session_factory, create_all

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


if __name__ == "__main__":
    asyncio.run(main())
