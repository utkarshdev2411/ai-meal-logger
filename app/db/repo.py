"""Typed data access. The only module allowed to write SQLAlchemy queries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy.sql import func

from app.db.models import LatencySample, Meal, MealItem, MealRevision, Memory, User


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Totals:
    kcal: float
    protein_g: float
    carbs_g: float
    fat_g: float
    fiber_g: float


async def get_or_create_user(session: AsyncSession, external_id: str, tz: str = "Asia/Kolkata") -> User:
    result = await session.execute(select(User).where(User.external_id == external_id))
    user = result.scalar_one_or_none()
    if user is not None:
        return user
    user = User(external_id=external_id, tz=tz, created_at=_now())
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user


async def insert_meal(
    session: AsyncSession,
    user_id: str,
    logged_at: datetime,
    local_date: date,
    meal_slot: str,
    source: str,
    items: list[dict[str, Any]],
    description: str | None = None,
    confidence: float | None = None,
    raw_input: str | None = None,
) -> Meal:
    now = _now()
    meal = Meal(
        user_id=user_id,
        logged_at=logged_at,
        local_date=local_date,
        meal_slot=meal_slot,
        description=description,
        source=source,
        confidence=confidence,
        raw_input=raw_input,
        created_at=now,
        updated_at=now,
        items=[MealItem(**item) for item in items],
    )
    session.add(meal)
    await session.commit()
    return await get_meal(session, meal.id)


async def get_meal(session: AsyncSession, meal_id: str) -> Meal | None:
    result = await session.execute(
        select(Meal).where(Meal.id == meal_id).options(selectinload(Meal.items))
    )
    return result.scalar_one_or_none()


async def last_active_meal(session: AsyncSession, user_id: str) -> Meal | None:
    # Ordered by created_at, not logged_at — meal_ref="last" means the thing
    # most recently *told to us*, per SCHEMA.md §1.2.
    result = await session.execute(
        select(Meal)
        .where(Meal.user_id == user_id, Meal.status == "active")
        .options(selectinload(Meal.items))
        .order_by(Meal.created_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def replace_meal_items(session: AsyncSession, meal_id: str, items: list[dict[str, Any]]) -> Meal:
    meal = await get_meal(session, meal_id)
    if meal is None:
        raise ValueError(f"no meal {meal_id}")
    for item in list(meal.items):
        await session.delete(item)
    meal.items = [MealItem(**item) for item in items]
    meal.updated_at = _now()
    await session.commit()
    return await get_meal(session, meal_id)


async def soft_delete_meal(session: AsyncSession, meal_id: str) -> None:
    meal = await get_meal(session, meal_id)
    if meal is None:
        raise ValueError(f"no meal {meal_id}")
    meal.status = "deleted"
    meal.updated_at = _now()
    await session.commit()


async def insert_revision(
    session: AsyncSession,
    meal_id: str,
    kind: str,
    before: Any = None,
    after: Any = None,
    reason: str | None = None,
) -> MealRevision:
    revision = MealRevision(
        meal_id=meal_id, kind=kind, before=before, after=after, reason=reason, created_at=_now()
    )
    session.add(revision)
    await session.commit()
    return revision


async def daily_totals(session: AsyncSession, user_id: str, day: date) -> Totals:
    stmt = (
        select(
            func.coalesce(func.sum(MealItem.kcal), 0),
            func.coalesce(func.sum(MealItem.protein_g), 0),
            func.coalesce(func.sum(MealItem.carbs_g), 0),
            func.coalesce(func.sum(MealItem.fat_g), 0),
            func.coalesce(func.sum(MealItem.fiber_g), 0),
        )
        .select_from(MealItem)
        .join(Meal, Meal.id == MealItem.meal_id)
        .where(
            Meal.user_id == user_id,
            Meal.local_date == day,
            Meal.status == "active",
        )
    )
    row = (await session.execute(stmt)).one()
    return Totals(*row)


async def recent_meals(session: AsyncSession, user_id: str, since_date: date) -> list[Meal]:
    result = await session.execute(
        select(Meal)
        .where(
            Meal.user_id == user_id,
            Meal.local_date >= since_date,
            Meal.status == "active",
        )
        .options(selectinload(Meal.items))
        .order_by(Meal.created_at.desc())
    )
    return list(result.scalars().all())


async def upsert_memory(
    session: AsyncSession,
    user_id: str,
    kind: str,
    key: str,
    value: Any,
    confidence: float,
    source_message: str | None = None,
) -> Memory:
    # Supersede-then-insert, one transaction: the old active row (if any) is
    # marked superseded and linked before the new active row commits, so the
    # partial unique index never sees two active rows at once.
    result = await session.execute(
        select(Memory).where(
            Memory.user_id == user_id,
            Memory.kind == kind,
            Memory.key == key,
            Memory.status == "active",
        )
    )
    existing = result.scalar_one_or_none()

    now = _now()
    new_memory = Memory(
        user_id=user_id,
        kind=kind,
        key=key,
        value=value,
        confidence=confidence,
        source_message=source_message,
        status="active",
        created_at=now,
        updated_at=now,
    )

    if existing is not None:
        existing.status = "superseded"
        existing.updated_at = now
        session.add(new_memory)
        await session.flush()
        existing.superseded_by = new_memory.id
    else:
        session.add(new_memory)

    await session.commit()
    return new_memory


async def get_active_memories(session: AsyncSession, user_id: str) -> list[Memory]:
    result = await session.execute(
        select(Memory).where(Memory.user_id == user_id, Memory.status == "active")
    )
    return list(result.scalars().all())


async def touch_memories(session: AsyncSession, memory_ids: list[str]) -> None:
    """Bump use_count/last_used_at for memories selected into a prompt (FR-5.7)."""
    if not memory_ids:
        return
    now = _now()
    result = await session.execute(select(Memory).where(Memory.id.in_(memory_ids)))
    for memory in result.scalars().all():
        memory.use_count += 1
        memory.last_used_at = now
    await session.commit()


async def insert_latency_sample(
    session: AsyncSession,
    turn_id: str,
    path: str,
    phase: str,
    ms: int,
    db_backend: str,
    user_id: str | None = None,
    fast_path: bool = False,
    cold: bool = False,
) -> LatencySample:
    sample = LatencySample(
        turn_id=turn_id,
        user_id=user_id,
        path=path,
        phase=phase,
        ms=ms,
        fast_path=fast_path,
        db_backend=db_backend,
        cold=cold,
        created_at=_now(),
    )
    session.add(sample)
    await session.commit()
    return sample
