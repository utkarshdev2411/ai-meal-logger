

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from app.db import repo
from app.db.client import async_session_factory
from app.db.models import Meal
from app.mealops.logging_ops import TotalsResult, get_totals
from app.memory.store import render_memory_block, retrieve_memories

# "last 2 days" per CONTEXT.md §8.2; capped separately so a binge-logging day
# can't blow up the prompt.
_DIGEST_DAYS = 1
_DIGEST_MAX_MEALS = 8


@dataclass
class Prefetch:
    memories: list[Any]
    totals: TotalsResult
    recent_meals: list[Meal]


async def prefetch(
    user_id: str,
    today: date | None = None,
    session_factory: Any = None,
) -> Prefetch:
    """Fetch memories, today's totals, and a recent-meal digest concurrently.

    A single `asyncio.gather` — not three sequential awaits — is the whole
    latency win this buys (CONTEXT.md §8.2's ~450-token / 1-2s-round-trip
    trade), so this shape must not regress back to sequential awaits.

    Each fetch gets its OWN session: an AsyncSession is not safe for
    concurrent use, and `retrieve_memories` writes (use_count/last_used_at),
    so sharing one session here raises IllegalStateChangeError rather than
    running in parallel.
    """
    today = today or date.today()
    since = today - timedelta(days=_DIGEST_DAYS)
    factory = session_factory or async_session_factory

    async def _in_session(fn):
        async with factory() as s:
            return await fn(s)

    memories, totals, recent = await asyncio.gather(
        _in_session(lambda s: retrieve_memories(s, user_id)),
        _in_session(lambda s: get_totals(s, user_id, day=today)),
        _in_session(lambda s: repo.recent_meals(s, user_id, since_date=since)),
    )
    return Prefetch(memories=memories, totals=totals, recent_meals=recent[:_DIGEST_MAX_MEALS])


def _fmt_num(value: float) -> str:
    return str(int(round(value)))


def _render_totals(totals: TotalsResult) -> str:
    t = totals.totals
    return (
        f"Today's totals so far: {_fmt_num(t.kcal)} kcal, {_fmt_num(t.protein_g)}g protein, "
        f"{_fmt_num(t.carbs_g)}g carbs, {_fmt_num(t.fat_g)}g fat, {_fmt_num(t.fiber_g)}g fiber"
    )


def _date_label(meal_date: date, today: date) -> str:
    delta = (today - meal_date).days
    if delta == 0:
        return "today"
    if delta == 1:
        return "yesterday"
    return meal_date.isoformat()


def _render_digest(recent_meals: list[Meal], today: date) -> str:
    if not recent_meals:
        return ""
    lines = [
        f"- {_date_label(m.local_date, today)} {m.meal_slot}: {m.description or '(no description)'}"
        for m in recent_meals
    ]
    return "Recent meals (last 2 days):\n" + "\n".join(lines)


def render_prefetch_block(fetched: Prefetch, today: date | None = None) -> str:
    """One combined block: memory facts + today's totals + recent-meal digest.

    Sections are separated blank-line-joined and any empty section is
    dropped, so an empty-history user gets a short block, not padded
    boilerplate.
    """
    today = today or date.today()
    sections = [
        render_memory_block(fetched.memories),
        _render_totals(fetched.totals),
        _render_digest(fetched.recent_meals, today),
    ]
    return "\n\n".join(s for s in sections if s)
