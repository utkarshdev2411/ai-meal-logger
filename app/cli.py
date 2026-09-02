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


import mimetypes
import uuid
from datetime import datetime, timezone
from app.db.models import Image
from app.vision.downscale import downscale_image

async def run_turn(checkpointer: MemorySaver, user_id: str, text: str, image_id: str | None = None) -> str:
    async with async_session_factory() as session:
        graph = build_graph(session, user_id, checkpointer)
        config = {"configurable": {"thread_id": THREAD_ID}}
        state_update = {"messages": [HumanMessage(content=text)], "user_id": user_id}
        if image_id:
            state_update["image_id"] = image_id
            
        result = await graph.ainvoke(state_update, config=config)
        
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

    await create_all()
    checkpointer = MemorySaver()
    async with async_session_factory() as session:
        user = await repo.get_or_create_user(session, DEMO_EXTERNAL_ID)
        user_id = user.id
        
        image_id = None
        if args.image:
            # Simulate upload and downscale
            with open(args.image, "rb") as f:
                raw_bytes = f.read()
            from app.config import get_settings
            max_edge = get_settings().image_max_edge
            down_bytes, w, h, mime = downscale_image(raw_bytes, max_edge)
            
            image_id = str(uuid.uuid4())
            # For CLI we can write down_bytes to a tmp file
            import tempfile
            tmp_path = tempfile.mktemp(suffix=".jpg")
            with open(tmp_path, "wb") as f:
                f.write(down_bytes)
                
            img_row = Image(
                id=image_id,
                user_id=user_id,
                path=tmp_path,
                mime=mime,
                width=w,
                height=h,
                bytes=len(down_bytes),
                status="pending",
                created_at=datetime.now(timezone.utc)
            )
            session.add(img_row)
            await session.commit()
            print(f"Image pre-processed and staged with ID: {image_id}")

    background_tasks: set[asyncio.Task] = set()
    print("CalorAI — type a message (Ctrl-D to quit)")
    
    first_turn = True
    while True:
        try:
            text = input("you> ").strip()
        except EOFError:
            print()
            break
        if not text and not (first_turn and image_id):
            continue
            
        turn_image_id = image_id if first_turn else None
        first_turn = False
        
        reply = await run_turn(checkpointer, user_id, text, turn_image_id)
        print(f"bot> {reply}")

        task = asyncio.create_task(_run_extraction(user_id, text, reply))
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)

    if background_tasks:
        await asyncio.gather(*background_tasks, return_exceptions=True)

if __name__ == "__main__":
    asyncio.run(main())
