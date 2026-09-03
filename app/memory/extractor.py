"""Background fact proposal (FR-5.5, FR-5.6, NFR-5.1, NFR-5.3).

Mirrors `app/nutrition/resolve.py::_call_llm_batch`'s httpx + `json_object` +
`model_validate` pattern deliberately, not a fresh design — same provider,
same graceful-failure shape, one fewer thing to review. `extract_and_write` is
the fire-and-forget entry point: it must never raise, since it's meant to be
handed to `asyncio.create_task` and walk away from.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from app.config import get_settings, next_api_key
from app.memory.store import write_memory

logger = logging.getLogger(__name__)

_LLM_TIMEOUT_S = 20.0
_WRITE_CONFIDENCE_FLOOR = 0.5

MemoryKind = Literal["diet", "goal", "routine", "alias", "preference", "dislike"]

_EXTRACTION_PROMPT = """You extract durable facts about a user from one turn of a meal-\
logging chat. Store a fact only if ALL of these hold:
- it is about the USER (not about this one meal)
- it would change how a FUTURE reply is written
- it is plausibly stable over weeks, not a one-off

NEVER propose a fact for: a specific meal that was logged (that's stored \
elsewhere), a transient state ("I'm full", "skipped lunch today"), anything \
derivable by looking at logged meals, or the raw message text itself.

Valid kinds: diet, goal, routine, alias, preference, dislike.

Reply with JSON only, matching this exact shape:
{{"facts": [{{"kind": "<diet|goal|routine|alias|preference|dislike>", \
"key": "<short_snake_case_key>", "value": {{"text": "<short fact text>"}}, \
"confidence": <0..1>}}]}}

If nothing durable was said, reply {{"facts": []}}.

User message: {user_message}
Assistant reply: {assistant_reply}"""


class FactCandidate(BaseModel):
    kind: MemoryKind
    key: str
    value: dict[str, Any]
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class FactBatch(BaseModel):
    facts: list[FactCandidate] = Field(default_factory=list)


async def extract_facts(
    session: Any,
    user_id: str,
    user_message: str,
    assistant_reply: str,
) -> list[dict[str, Any]]:
    """Calls EXTRACTOR_MODEL and returns candidate facts. Never raises —
    a failure yields an empty list, same fallback shape as nutrition's."""
    settings = get_settings()
    try:
        api_key = next_api_key()
    except Exception:
        return []

    prompt = _EXTRACTION_PROMPT.format(user_message=user_message, assistant_reply=assistant_reply)

    try:
        async with httpx.AsyncClient(
            base_url=settings.llm_base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=_LLM_TIMEOUT_S,
        ) as client:
            response = await client.post(
                "/chat/completions",
                json={
                    "model": settings.extractor_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 400,
                    "response_format": {"type": "json_object"},
                },
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        batch = FactBatch.model_validate(json.loads(content))
    except Exception:  # network, auth, malformed JSON — never propagate (NFR-5.3)
        logger.warning("memory extractor call failed for message %r", user_message, exc_info=True)
        return []

    return [f.model_dump() for f in batch.facts]


async def extract_and_write(
    session: Any,
    user_id: str,
    user_message: str,
    assistant_reply: str,
) -> None:
    """Fire-and-forget: propose facts, write the ones above the confidence
    floor. Must never raise into the caller (NFR-5.1, NFR-5.3) — it's meant to
    run detached via `asyncio.create_task`."""
    try:
        candidates = await extract_facts(session, user_id, user_message, assistant_reply)
        for fact in candidates:
            if fact.get("confidence", 0.0) < _WRITE_CONFIDENCE_FLOOR:
                continue
            await write_memory(
                session,
                user_id=user_id,
                kind=fact["kind"],
                key=fact["key"],
                value=fact["value"],
                confidence=fact["confidence"],
                source_message=user_message,
            )
    except Exception:
        logger.warning("extract_and_write failed for user %s", user_id, exc_info=True)
