import asyncio
import json
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, File, Header, Request, UploadFile
from fastapi.responses import HTMLResponse
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessageChunk, HumanMessage
from sse_starlette.sse import EventSourceResponse

from app.agent.graph import build_graph
from app.config import get_settings
from app.db.client import async_session_factory, create_all
from app.db import repo
from app.db.models import Image
from app.memory.extractor import extract_and_write
from app.vision.downscale import downscale_image
from app.vision.process import process_image_by_id

DEMO_USER = "demo"
_checkpointer = MemorySaver()

app = FastAPI(title="CalorAI")

STATIC_DIR = Path(__file__).parent / "static"


@app.on_event("startup")
async def startup():
    await create_all()


@app.get("/", response_class=HTMLResponse)
async def index():
    html_file = STATIC_DIR / "index.html"
    return HTMLResponse(content=html_file.read_text())


def _resolve_user_header(x_user_id: str | None) -> str:
    return x_user_id.strip() if x_user_id and x_user_id.strip() else DEMO_USER


@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    x_user_id: str | None = Header(default=None),
):
    external_id = _resolve_user_header(x_user_id)
    raw = await file.read()
    settings = get_settings()
    down_bytes, w, h, mime = downscale_image(raw, settings.image_max_edge)

    image_id = str(uuid.uuid4())
    import tempfile
    tmp = tempfile.mktemp(suffix=".jpg")
    with open(tmp, "wb") as f:
        f.write(down_bytes)

    async with async_session_factory() as session:
        user = await repo.get_or_create_user(session, external_id)
        img = Image(
            id=image_id,
            user_id=user.id,
            path=tmp,
            mime=mime,
            width=w,
            height=h,
            bytes=len(down_bytes),
            status="pending",
            created_at=datetime.now(timezone.utc),
        )
        session.add(img)
        await session.commit()

    # kick off prewarm in background — do NOT await
    asyncio.create_task(_prewarm(image_id))

    return {"image_id": image_id, "width": w, "height": h}


async def _prewarm(image_id: str) -> None:
    try:
        async with async_session_factory() as session:
            await process_image_by_id(session, image_id)
    except Exception:
        pass


@app.post("/chat")
async def chat(
    request: Request,
    x_user_id: str | None = Header(default=None),
):
    body = await request.json()
    text = (body.get("text") or "").strip()
    image_id = body.get("image_id")
    thread_id = body.get("thread_id", "default")
    external_id = _resolve_user_header(x_user_id)

    async with async_session_factory() as session:
        user = await repo.get_or_create_user(session, external_id)
        user_id = user.id

    async def event_stream() -> AsyncGenerator[dict, None]:
        state_update: dict = {
            "messages": [HumanMessage(content=text or "(image attached)")],
            "user_id": user_id,
        }
        if image_id:
            state_update["image_id"] = image_id

        config = {"configurable": {"thread_id": f"{user_id}:{thread_id}"}}
        full_reply = ""

        async with async_session_factory() as session:
            graph = build_graph(session, user_id, _checkpointer)

            async for event in graph.astream_events(state_update, config=config, version="v2"):
                kind = event.get("event")

                # stream token deltas
                if kind == "on_chat_model_stream":
                    chunk = event["data"].get("chunk")
                    if isinstance(chunk, AIMessageChunk) and chunk.content:
                        token = chunk.content if isinstance(chunk.content, str) else ""
                        if token:
                            full_reply += token
                            yield {"event": "token", "data": json.dumps({"token": token})}

                # emit meal_logged event as soon as tool returns
                elif kind == "on_tool_end":
                    tool_name = event.get("name", "")
                    if tool_name == "log_meal":
                        output = event["data"].get("output", {})
                        if isinstance(output, dict):
                            yield {
                                "event": "meal_logged",
                                "data": json.dumps(output),
                            }

            yield {"event": "done", "data": json.dumps({"reply": full_reply})}

        # fire-and-forget memory extraction
        asyncio.create_task(_extract(user_id, text, full_reply))

    return EventSourceResponse(event_stream())


async def _extract(user_id: str, text: str, reply: str) -> None:
    try:
        async with async_session_factory() as session:
            await extract_and_write(session, user_id, text, reply)
    except Exception:
        pass


@app.get("/totals")
async def get_totals(
    day: str | None = None,
    x_user_id: str | None = Header(default=None),
):
    external_id = _resolve_user_header(x_user_id)
    target_date = date.fromisoformat(day) if day else date.today()
    async with async_session_factory() as session:
        user = await repo.get_or_create_user(session, external_id)
        totals = await repo.daily_totals(session, user.id, target_date)
    return {
        "date": target_date.isoformat(),
        "kcal": totals.kcal,
        "protein_g": totals.protein_g,
        "carbs_g": totals.carbs_g,
        "fat_g": totals.fat_g,
        "fiber_g": totals.fiber_g,
    }


@app.get("/meals")
async def get_meals(
    since: str | None = None,
    x_user_id: str | None = Header(default=None),
):
    external_id = _resolve_user_header(x_user_id)
    since_date = date.fromisoformat(since) if since else date.today()
    async with async_session_factory() as session:
        user = await repo.get_or_create_user(session, external_id)
        meals = await repo.recent_meals(session, user.id, since_date)

    result = []
    for m in meals:
        result.append({
            "id": m.id,
            "meal_slot": m.meal_slot,
            "logged_at": m.logged_at.isoformat() if m.logged_at else None,
            "local_date": m.local_date.isoformat() if m.local_date else None,
            "description": m.description,
            "status": m.status,
            "items": [
                {
                    "id": item.id,
                    "name": item.name,
                    "quantity": item.quantity,
                    "unit": item.unit,
                    "kcal": item.kcal,
                    "protein_g": item.protein_g,
                    "carbs_g": item.carbs_g,
                    "fat_g": item.fat_g,
                }
                for item in m.items
            ],
        })
    return result
