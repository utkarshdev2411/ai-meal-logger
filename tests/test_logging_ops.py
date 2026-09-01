"""Correctness tests for the logging & totals engine (PHASES.md Phase 3).

Everything here uses only foods resolvable purely from `NUTRITION_TABLE`
(roti, paratha, chai, rice, dal, ...) so `nutrition.resolve()` never reaches
the LLM fallback — the whole suite runs offline, no API key, no network.

Each test gets a fresh in-memory SQLite engine (StaticPool keeps the one
:memory: connection alive for the session's lifetime) so tests share no state.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy import func, select

from app.db import repo
from app.db.models import Base, Meal, MealRevision
from app.mealops.logging_ops import MealLoggingError, log_meal, revise_meal, get_totals

ROTI = {"name": "roti", "kcal": 71, "protein_g": 3.0, "carbs_g": 15.0, "fat_g": 0.4, "fiber_g": 2.0}
CHAI = {"name": "chai", "kcal": 60, "protein_g": 2.0, "carbs_g": 8.0, "fat_g": 2.0, "fiber_g": 0.0}
PARATHA = {"name": "paratha", "kcal": 126, "protein_g": 3.0, "carbs_g": 18.0, "fat_g": 5.0, "fiber_g": 2.0}
DAL = {"name": "dal", "kcal": 180, "protein_g": 9.0, "carbs_g": 27.0, "fat_g": 4.0, "fiber_g": 6.0}

TODAY = datetime(2026, 9, 2, 9, 0, tzinfo=timezone.utc)


@pytest.fixture
async def session():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


async def make_user(session, external_id: str = "u1") -> str:
    user = await repo.get_or_create_user(session, external_id)
    return user.id


def qty(food: dict, quantity: float) -> dict:
    return {"name": food["name"], "quantity": quantity}


async def revision_count(session, meal_id: str) -> int:
    result = await session.execute(
        select(func.count()).select_from(MealRevision).where(MealRevision.meal_id == meal_id)
    )
    return result.scalar_one()


async def meal_row_count(session) -> int:
    result = await session.execute(select(func.count()).select_from(Meal))
    return result.scalar_one()


# 1. log_meal -> get_totals matches the sum of logged items exactly.
async def test_log_meal_matches_totals(session):
    user_id = await make_user(session)
    result = await log_meal(session, user_id, [qty(ROTI, 3), qty(CHAI, 1)], "breakfast", TODAY, "text")

    expected_kcal = ROTI["kcal"] * 3 + CHAI["kcal"] * 1
    assert result.daily_totals.kcal == pytest.approx(expected_kcal)
    assert result.meal.totals.kcal == pytest.approx(expected_kcal)

    totals = await get_totals(session, user_id, day=TODAY.date())
    assert totals.totals.kcal == pytest.approx(expected_kcal)


# 2. The critical regression: qty correction does not double count.
async def test_set_item_qty_does_not_double_count(session):
    user_id = await make_user(session)
    logged = await log_meal(session, user_id, [qty(ROTI, 2)], "dinner", TODAY, "text")
    item_id = logged.meal.items[0].id

    revised = await revise_meal(
        session, user_id, "last", "set_item_qty", item_id=item_id, quantity=3, reason="actually 3 rotis"
    )

    assert revised.meal.totals.kcal == pytest.approx(ROTI["kcal"] * 3)
    assert revised.daily_totals.kcal == pytest.approx(ROTI["kcal"] * 3)
    assert await meal_row_count(session) == 1


# 3. remove_item / add_item / replace_items each mutate the same meal.
async def test_remove_item(session):
    user_id = await make_user(session)
    logged = await log_meal(session, user_id, [qty(ROTI, 2), qty(CHAI, 1)], "breakfast", TODAY, "text")
    roti_item = next(i for i in logged.meal.items if i.name == "roti")

    revised = await revise_meal(session, user_id, "last", "remove_item", item_id=roti_item.id)

    assert len(revised.meal.items) == 1
    assert revised.meal.items[0].name == "chai"
    assert revised.daily_totals.kcal == pytest.approx(CHAI["kcal"])
    assert await meal_row_count(session) == 1


async def test_add_item(session):
    user_id = await make_user(session)
    await log_meal(session, user_id, [qty(ROTI, 2)], "breakfast", TODAY, "text")

    revised = await revise_meal(session, user_id, "last", "add_item", name="chai", quantity=1)

    assert len(revised.meal.items) == 2
    assert revised.daily_totals.kcal == pytest.approx(ROTI["kcal"] * 2 + CHAI["kcal"])
    assert await meal_row_count(session) == 1


async def test_replace_items(session):
    user_id = await make_user(session)
    await log_meal(session, user_id, [qty(ROTI, 2), qty(CHAI, 1)], "breakfast", TODAY, "text")

    revised = await revise_meal(
        session, user_id, "last", "replace_items", items=[qty(PARATHA, 2), qty(DAL, 1)]
    )

    assert {i.name for i in revised.meal.items} == {"paratha", "dal"}
    assert revised.daily_totals.kcal == pytest.approx(PARATHA["kcal"] * 2 + DAL["kcal"])
    assert await meal_row_count(session) == 1


# 4. delete_meal -> totals zero, row still exists as status='deleted'.
async def test_delete_meal_soft_deletes(session):
    user_id = await make_user(session)
    logged = await log_meal(session, user_id, [qty(ROTI, 2)], "breakfast", TODAY, "text")

    revised = await revise_meal(session, user_id, "last", "delete_meal", reason="didn't eat it")

    assert revised.daily_totals.kcal == 0
    meal_row = await repo.get_meal(session, logged.meal.meal_id)
    assert meal_row is not None
    assert meal_row.status == "deleted"


# 5. A deleted meal never reappears in totals or breakdown.
async def test_deleted_meal_excluded_from_totals_and_breakdown(session):
    user_id = await make_user(session)
    logged = await log_meal(session, user_id, [qty(ROTI, 2)], "breakfast", TODAY, "text")
    await revise_meal(session, user_id, "last", "delete_meal")

    totals = await get_totals(session, user_id, day=TODAY.date())
    assert totals.totals.kcal == 0
    assert logged.meal.meal_id not in {m.meal_id for m in totals.meals}


# 6. Every mutation produces exactly one new meal_revisions row.
async def test_revision_count_matches_mutation_count(session):
    user_id = await make_user(session)
    logged = await log_meal(session, user_id, [qty(ROTI, 2)], "breakfast", TODAY, "text")
    meal_id = logged.meal.meal_id

    after_add = await revise_meal(session, user_id, "last", "add_item", name="chai", quantity=1)
    roti_item = next(i for i in after_add.meal.items if i.name == "roti")
    await revise_meal(session, user_id, "last", "set_item_qty", item_id=roti_item.id, quantity=4)
    await revise_meal(session, user_id, "last", "delete_meal")

    assert await revision_count(session, meal_id) == 3


# 7. meal_ref="last" picks the most recently created active meal, ignoring deleted ones.
async def test_last_picks_most_recently_created_active_meal(session):
    user_id = await make_user(session)
    first = await log_meal(session, user_id, [qty(ROTI, 1)], "breakfast", TODAY, "text")
    second = await log_meal(session, user_id, [qty(CHAI, 1)], "breakfast", TODAY, "text")
    third = await log_meal(session, user_id, [qty(DAL, 1)], "lunch", TODAY, "text")

    await revise_meal(session, user_id, "last", "delete_meal")  # deletes `third`

    revised = await revise_meal(session, user_id, "last", "remove_item", item_id=second.meal.items[0].id)
    assert revised.meal.meal_id == second.meal.meal_id

    third_row = await repo.get_meal(session, third.meal.meal_id)
    assert third_row.status == "deleted"


# 8. Two different user_ids have fully isolated totals.
async def test_user_isolation(session):
    user_a = await make_user(session, "alice")
    user_b = await make_user(session, "bob")

    await log_meal(session, user_a, [qty(ROTI, 2)], "breakfast", TODAY, "text")
    await log_meal(session, user_b, [qty(PARATHA, 1)], "breakfast", TODAY, "text")

    totals_a = await get_totals(session, user_a, day=TODAY.date())
    totals_b = await get_totals(session, user_b, day=TODAY.date())

    assert totals_a.totals.kcal == pytest.approx(ROTI["kcal"] * 2)
    assert totals_b.totals.kcal == pytest.approx(PARATHA["kcal"])


# 9. A meal logged for a different local_date doesn't appear in today's totals.
async def test_different_local_date_excluded(session):
    user_id = await make_user(session)
    yesterday = TODAY - timedelta(days=1)
    await log_meal(session, user_id, [qty(ROTI, 2)], "breakfast", yesterday, "text")
    await log_meal(session, user_id, [qty(CHAI, 1)], "breakfast", TODAY, "text")

    totals = await get_totals(session, user_id, day=TODAY.date())
    assert totals.totals.kcal == pytest.approx(CHAI["kcal"])
    assert len(totals.meals) == 1


# 10. revise_meal(meal_ref="last") with no meals raises a clear error.
async def test_revise_last_with_no_meals_raises(session):
    user_id = await make_user(session)
    with pytest.raises(MealLoggingError):
        await revise_meal(session, user_id, "last", "delete_meal")


# 11. Example-based sequences: totals always match an independently
# recomputed reference sum, across several corrections in a row.
async def test_sequence_matches_reference_sum(session):
    user_id = await make_user(session)
    reference: dict[str, float] = {}

    def add(name: str, kcal_each: float, quantity: float) -> None:
        reference[name] = reference.get(name, 0.0) + kcal_each * quantity

    def total() -> float:
        return sum(reference.values())

    logged = await log_meal(session, user_id, [qty(ROTI, 2), qty(CHAI, 1)], "breakfast", TODAY, "text")
    add("roti", ROTI["kcal"], 2)
    add("chai", CHAI["kcal"], 1)
    totals = await get_totals(session, user_id, day=TODAY.date())
    assert totals.totals.kcal == pytest.approx(total())

    roti_item = next(i for i in logged.meal.items if i.name == "roti")
    revised = await revise_meal(session, user_id, "last", "set_item_qty", item_id=roti_item.id, quantity=5)
    reference["roti"] = ROTI["kcal"] * 5
    totals = await get_totals(session, user_id, day=TODAY.date())
    assert totals.totals.kcal == pytest.approx(total())

    revised = await revise_meal(session, user_id, "last", "add_item", name="dal", quantity=1)
    add("dal", DAL["kcal"], 1)
    totals = await get_totals(session, user_id, day=TODAY.date())
    assert totals.totals.kcal == pytest.approx(total())

    chai_item = next(i for i in revised.meal.items if i.name == "chai")
    revised = await revise_meal(session, user_id, "last", "remove_item", item_id=chai_item.id)
    del reference["chai"]
    totals = await get_totals(session, user_id, day=TODAY.date())
    assert totals.totals.kcal == pytest.approx(total())

    await revise_meal(session, user_id, "last", "delete_meal")
    reference.clear()
    totals = await get_totals(session, user_id, day=TODAY.date())
    assert totals.totals.kcal == pytest.approx(total())
    assert totals.totals.kcal == 0

    assert await meal_row_count(session) == 1
