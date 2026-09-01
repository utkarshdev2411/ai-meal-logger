"""Write / rank / budget / retrieve durable facts (CONTEXT.md §6, FR-5.7).

Two callers write through `write_memory` — the explicit `remember` tool and
the background extractor — but neither the storage rule nor the never-store
list is enforced here. That judgement belongs to whichever caller decided the
fact is worth persisting; this module's job is just the persist-and-rank
mechanics, so it stays usable (and testable) by both write paths unchanged.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.db import repo
from app.db.models import Memory

# Higher-priority kinds win ties and are worth keeping in a tight budget first:
# diet/goal shape almost every reply (nutrition assumptions, progress math),
# routine/alias resolve "my usual" for free, preference/dislike are the long tail.
_KIND_PRIORITY: dict[str, float] = {
    "diet": 1.0,
    "goal": 0.9,
    "routine": 0.8,
    "alias": 0.8,
    "preference": 0.6,
    "dislike": 0.6,
}
_DEFAULT_KIND_PRIORITY = 0.5

_HALF_LIFE_DAYS = 14.0  # recency decay: a fact unused for ~2 weeks is worth half


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _render_fact(memory: Memory) -> str:
    value = memory.value
    if isinstance(value, dict):
        text = value.get("text") or value.get("label") or value.get("phrase")
        if text:
            return str(text)
        if memory.kind == "goal":
            metric = value.get("metric", memory.key)
            target = value.get("target")
            period = value.get("period", "day")
            return f"goal: {target} {metric}/{period}" if target is not None else f"goal: {metric}"
        if memory.kind == "routine":
            items = value.get("items") or []
            items_txt = ", ".join(
                f"{i.get('quantity', '')} {i.get('unit', '')} {i.get('name', '')}".strip()
                for i in items
            )
            label = value.get("label", memory.key)
            return f"{label}: {items_txt}" if items_txt else label
    return f"{memory.key}: {value}"


def _recency_score(memory: Memory) -> float:
    reference = memory.last_used_at or memory.created_at
    if reference is None:
        return 1.0
    now = datetime.now(timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    age_days = max(0.0, (now - reference).total_seconds() / 86400)
    return 0.5 ** (age_days / _HALF_LIFE_DAYS)


def _rank_score(memory: Memory) -> float:
    kind_priority = _KIND_PRIORITY.get(memory.kind, _DEFAULT_KIND_PRIORITY)
    recency = _recency_score(memory)
    use_count_bonus = 1.0 + min(memory.use_count, 10) * 0.05
    return kind_priority * recency * use_count_bonus


async def write_memory(
    session: Any,
    user_id: str,
    kind: str,
    key: str,
    value: Any,
    confidence: float,
    source_message: str | None = None,
) -> Memory:
    return await repo.upsert_memory(
        session,
        user_id=user_id,
        kind=kind,
        key=key,
        value=value,
        confidence=confidence,
        source_message=source_message,
    )


async def retrieve_memories(
    session: Any,
    user_id: str,
    top_k: int | None = None,
    token_budget: int | None = None,
) -> list[Memory]:
    """Rank active memories by kind_priority x recency x use_count, take the
    best that fit both `top_k` and `token_budget` (NFR-5.2), and bump
    use_count/last_used_at on the ones actually selected."""
    from app.config import get_settings

    settings = get_settings()
    top_k = top_k if top_k is not None else settings.memory_top_k
    token_budget = token_budget if token_budget is not None else settings.memory_token_budget

    candidates = await repo.get_active_memories(session, user_id)
    candidates.sort(key=_rank_score, reverse=True)

    selected: list[Memory] = []
    used_tokens = 0
    for memory in candidates[:top_k]:
        line_tokens = _estimate_tokens(_render_fact(memory))
        if selected and used_tokens + line_tokens > token_budget:
            break
        selected.append(memory)
        used_tokens += line_tokens

    if selected:
        await repo.touch_memories(session, [m.id for m in selected])
    return selected


def render_memory_block(memories: list[Memory]) -> str:
    if not memories:
        return ""
    lines = "\n".join(f"- {_render_fact(m)}" for m in memories)
    return f"Known facts about this user:\n{lines}"
