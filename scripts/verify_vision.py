"""Verification script for vision and prefetch paths."""


import asyncio
import io
import json
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SCRATCH_DB = tempfile.mktemp(suffix=".sqlite3")
os.environ.setdefault("LLM_API_KEY", "test-key-not-used")
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{SCRATCH_DB}"

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from PIL import Image as PILImage  # noqa: E402

from app.agent.graph import build_graph  # noqa: E402
from app.agent.prefetch import prefetch, render_prefetch_block  # noqa: E402
from app.db import repo  # noqa: E402
from app.db.client import async_session_factory, create_all  # noqa: E402
from app.db.models import Image  # noqa: E402
from app.mealops.logging_ops import log_meal  # noqa: E402
from app.vision.downscale import downscale_image  # noqa: E402
from app.vision.extract import extract_vision  # noqa: E402
from app.vision.process import process_image_by_id  # noqa: E402

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'} — {label}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(label)


def _png(w: int, h: int) -> bytes:
    buf = io.BytesIO()
    PILImage.new("RGB", (w, h), (120, 90, 60)).save(buf, format="PNG")
    return buf.getvalue()


GOOD_OBS = {
    "items": [
        {"name": "dal", "portion_estimate": "1 bowl", "confidence": 0.9, "alternatives": []},
        {"name": "rice", "portion_estimate": "1 cup", "confidence": 0.85, "alternatives": []},
    ],
    "plate_context": "thali",
    "overall_confidence": 0.87,
    "unclear": [],
}


def _reply(content: str):
    """Fake an OpenAI-compatible chat response carrying `content`."""
    resp = AsyncMock()
    resp.raise_for_status = lambda: None
    resp.json = lambda: {"choices": [{"message": {"content": content}}]}
    return resp


class ScriptedLLM:
    """Duck-typed stand-in: replays canned turns, records the prompt it saw."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.last_prompt = None
        self.tool_calls_seen: list[str] = []

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, prompt, **kwargs):
        self.last_prompt = prompt
        msg = self.turns.pop(0)
        for tc in getattr(msg, "tool_calls", None) or []:
            self.tool_calls_seen.append(tc["name"])
        return msg


async def main() -> None:
    await create_all()

    # --- downscale -------------------------------------------------------
    raw = _png(2000, 1000)
    small, w, h, mime = downscale_image(raw, 768)
    check("downscale caps the long edge and preserves aspect", (w, h) == (768, 384), f"{w}x{h} {mime}")
    check("downscale shrinks the payload", len(small) < len(raw), f"{len(raw)} -> {len(small)} bytes")

    # --- extract: json_object parse, fence tolerance, retry, degrade -----
    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=_reply(json.dumps(GOOD_OBS)))):
        obs = await extract_vision(small, mime)
    check("extract parses a plain json_object response", [i.name for i in obs.items] == ["dal", "rice"])

    fenced = "```json\n" + json.dumps(GOOD_OBS) + "\n```"
    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=_reply(fenced))):
        obs2 = await extract_vision(small, mime)
    check("extract tolerates markdown-fenced JSON", obs2.overall_confidence == 0.87)

    post = AsyncMock(side_effect=[_reply("not json at all"), _reply(json.dumps(GOOD_OBS))])
    with patch("httpx.AsyncClient.post", new=post):
        obs3 = await extract_vision(small, mime)
    check("malformed JSON retries once and then succeeds", post.await_count == 2 and len(obs3.items) == 2)

    post_bad = AsyncMock(return_value=_reply("still not json"))
    raised = False
    with patch("httpx.AsyncClient.post", new=post_bad):
        try:
            await extract_vision(small, mime)
        except Exception:
            raised = True
    check("two bad responses raise rather than fabricate", raised and post_bad.await_count == 2)

    # --- process_image_by_id: caches, and degrades instead of raising ----
    async with async_session_factory() as session:
        user = await repo.get_or_create_user(session, "vision-demo")
        user_id = user.id

        img_path = tempfile.mktemp(suffix=".png")
        Path(img_path).write_bytes(small)
        img = Image(user_id=user_id, path=img_path, mime=mime, status="pending",
                    created_at=datetime.now().astimezone())
        session.add(img)
        await session.commit()
        image_id = img.id

    with patch("app.vision.process.extract_vision", new=AsyncMock(return_value=obs)):
        async with async_session_factory() as session:
            first = await process_image_by_id(session, image_id)
    check("process_image_by_id stores an observation", "items" in first)

    inner = AsyncMock(return_value=obs)
    with patch("app.vision.process.extract_vision", new=inner):
        async with async_session_factory() as session:
            cached = await process_image_by_id(session, image_id)
    check("a ready image is served from cache, vision not re-called",
          inner.await_count == 0 and cached["items"][0]["name"] == "dal")

    async with async_session_factory() as session:
        bad = Image(user_id=user_id, path="/nonexistent/nope.png", mime=mime, status="pending",
                    created_at=datetime.now().astimezone())
        session.add(bad)
        await session.commit()
        bad_id = bad.id
    async with async_session_factory() as session:
        failed = await process_image_by_id(session, bad_id)
    check("an unreadable image degrades to an error dict, no exception", "error" in failed)

    # --- graph: a failed extraction must not reach the prompt as data ----
    async with async_session_factory() as session:
        llm = ScriptedLLM([AIMessage(content="Couldn't read that one — what was on the plate?")])
        graph = build_graph(session, user_id, MemorySaver(), llm=llm)
        await graph.ainvoke(
            {"messages": [HumanMessage(content="")], "user_id": user_id, "image_id": bad_id},
            config={"configurable": {"thread_id": "vision-fail"}},
        )
    sys_text = next(m.content for m in llm.last_prompt if isinstance(m, SystemMessage))
    check("vision failure becomes a note, not a pasted error payload",
          "VISION_NOTE" in sys_text and "VISION_OBSERVATION" not in sys_text)

    # --- graph: photo + caption -> exactly ONE meal (graded regression) --
    async with async_session_factory() as session:
        ok_path = tempfile.mktemp(suffix=".png")
        Path(ok_path).write_bytes(small)
        good = Image(user_id=user_id, path=ok_path, mime=mime, status="ready",
                     observation=GOOD_OBS, created_at=datetime.now().astimezone())
        session.add(good)
        await session.commit()
        good_id = good.id

    log_call = AIMessage(
        content="",
        tool_calls=[{
            "name": "log_meal",
            "args": {"items": [{"name": "dal", "quantity": 0.5},
                               {"name": "rice", "quantity": 0.5}],
                     "meal_slot": "dinner"},
            "id": "call_1",
        }],
    )
    async with async_session_factory() as session:
        llm = ScriptedLLM([log_call, AIMessage(content="Logged half a thali — dal and rice.")])
        graph = build_graph(session, user_id, MemorySaver(), llm=llm)
        await graph.ainvoke(
            {"messages": [HumanMessage(content="half of this was my brother's")],
             "user_id": user_id, "image_id": good_id},
            config={"configurable": {"thread_id": "vision-merge"}},
        )
    sys_text = next(m.content for m in llm.last_prompt if isinstance(m, SystemMessage))
    async with async_session_factory() as session:
        meals = await repo.recent_meals(session, user_id, since_date=datetime.now().date())
    check("photo + caption resolve to exactly ONE meal",
          len(meals) == 1, f"{len(meals)} meal(s), tools={llm.tool_calls_seen}")
    check("the caption and the observation share one agent turn",
          "VISION_OBSERVATION" in sys_text and llm.tool_calls_seen.count("log_meal") == 1)

    # --- prefetch block + zero-tool-call query --------------------------
    async with async_session_factory() as session:
        await repo.upsert_memory(session, user_id, "diet", "vegetarian",
                                 {"text": "vegetarian"}, 1.0, "i'm vegetarian btw")
        await log_meal(session, user_id, [{"name": "poha", "quantity": 1}], "breakfast",
                       (datetime.now() - timedelta(days=1)).astimezone(), "text")

    fetched = await prefetch(user_id)
    block = render_prefetch_block(fetched)
    check("prefetch block carries facts, today's totals, and yesterday's meals",
          "vegetarian" in block and "Today's totals" in block and "yesterday" in block)

    async with async_session_factory() as session:
        llm = ScriptedLLM([AIMessage(content="You're at 438 kcal today.")])
        graph = build_graph(session, user_id, MemorySaver(), llm=llm)
        await graph.ainvoke(
            {"messages": [HumanMessage(content="how am I doing on calories?")], "user_id": user_id},
            config={"configurable": {"thread_id": "prefetch-query"}},
        )
    check("a today-totals question costs ZERO tool calls",
          llm.tool_calls_seen == [], f"tools={llm.tool_calls_seen}")

    sys_text = next(m.content for m in llm.last_prompt if isinstance(m, SystemMessage))
    check("static policy text precedes the volatile prefetch block",
          sys_text.index("Ambiguity policy") < sys_text.index("Today's totals"))

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED: " + "; ".join(failures))
        sys.exit(1)
    print("verify_vision.py: all checks passed")


if __name__ == "__main__":
    asyncio.run(main())
