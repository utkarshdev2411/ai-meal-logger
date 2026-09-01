"""Throwaway verification of the persistence layer against a scratch SQLite file.

Not a test suite (per the brief) — a script proving the correctness-critical
paths: no double-count on revision, soft-delete zeroes totals, memory
supersede, and the partial unique index enforced at the DB level.
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import date, datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DB_PATH = "/tmp/verify_db_scratch.sqlite3"
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)
os.environ.setdefault("LLM_API_KEY", "test-key-not-used")
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{DB_PATH}"

from sqlalchemy.exc import IntegrityError  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402

from app.db import repo  # noqa: E402
from app.db.models import Base, Memory  # noqa: E402

results: list[tuple[str, bool]] = []


def check(name: str, condition: bool) -> None:
    results.append((name, condition))
    print(f"{'PASS' if condition else 'FAIL'} — {name}")


async def main() -> None:
    engine = create_async_engine(os.environ["DATABASE_URL"])
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    check("create_all runs without error", True)

    async with session_factory() as session:
        user = await repo.get_or_create_user(session, external_id="verify-user")
        check("user created", user.id is not None)

        today = date.today()
        now = datetime.now(timezone.utc)
        items = [
            dict(name="paratha", canonical_key="paratha", quantity=2, unit="piece",
                 kcal=300, protein_g=8, carbs_g=40, fat_g=12, fiber_g=3,
                 nutrition_source="table", confidence=0.9),
            dict(name="chai", canonical_key="chai", quantity=1, unit="cup",
                 kcal=80, protein_g=2, carbs_g=10, fat_g=3, fiber_g=0,
                 nutrition_source="table", confidence=0.9),
        ]
        meal = await repo.insert_meal(
            session, user_id=user.id, logged_at=now, local_date=today,
            meal_slot="breakfast", source="text", items=items,
            description="2 parathas + chai", raw_input="2 parathas and chai",
        )
        totals = await repo.daily_totals(session, user.id, today)
        check("meal logged with 2 items", len(meal.items) == 2)
        check("daily_totals matches sum after log", totals.kcal == 380)

        before_snapshot = [
            {"name": i.name, "quantity": float(i.quantity), "kcal": float(i.kcal)}
            for i in meal.items
        ]
        new_items = [
            dict(name="paratha", canonical_key="paratha", quantity=3, unit="piece",
                 kcal=450, protein_g=12, carbs_g=60, fat_g=18, fiber_g=4.5,
                 nutrition_source="table", confidence=0.9),
            dict(name="chai", canonical_key="chai", quantity=1, unit="cup",
                 kcal=80, protein_g=2, carbs_g=10, fat_g=3, fiber_g=0,
                 nutrition_source="table", confidence=0.9),
        ]
        revised = await repo.replace_meal_items(session, meal.id, new_items)
        after_snapshot = [
            {"name": i.name, "quantity": float(i.quantity), "kcal": float(i.kcal)}
            for i in revised.items
        ]
        await repo.insert_revision(
            session, meal_id=meal.id, kind="set_item_qty",
            before=before_snapshot, after=after_snapshot,
            reason="actually that was 3 rotis not 2",
        )
        totals_after_revision = await repo.daily_totals(session, user.id, today)
        check(
            "no double-count: revised total is 530, not 380+530",
            totals_after_revision.kcal == 530,
        )

        last = await repo.last_active_meal(session, user.id)
        check("last_active_meal resolves to the revised meal", last is not None and last.id == meal.id)

        await repo.soft_delete_meal(session, meal.id)
        totals_after_delete = await repo.daily_totals(session, user.id, today)
        check("soft-delete zeroes totals", totals_after_delete.kcal == 0)

        mem1 = await repo.upsert_memory(
            session, user_id=user.id, kind="goal", key="protein_target",
            value={"metric": "protein_g", "target": 120, "period": "day"},
            confidence=1.0, source_message="target 120g protein",
        )
        mem2 = await repo.upsert_memory(
            session, user_id=user.id, kind="goal", key="protein_target",
            value={"metric": "protein_g", "target": 140, "period": "day"},
            confidence=1.0, source_message="actually make it 140g protein",
        )
        active = await repo.get_active_memories(session, user.id)
        await session.refresh(mem1)
        check("exactly one active memory row for the key", len(active) == 1 and active[0].id == mem2.id)
        check("old memory row marked superseded", mem1.status == "superseded" and mem1.superseded_by == mem2.id)

        integrity_violation = False
        try:
            dupe = Memory(
                user_id=user.id, kind="goal", key="protein_target",
                value={"metric": "protein_g", "target": 999, "period": "day"},
                confidence=1.0, status="active",
                created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc),
            )
            session.add(dupe)
            await session.commit()
        except IntegrityError:
            integrity_violation = True
            await session.rollback()
        check("partial unique index rejects a second active row at the DB level", integrity_violation)

    await engine.dispose()

    failed = [name for name, ok in results if not ok]
    print()
    if failed:
        print(f"{len(failed)} check(s) FAILED: {failed}")
        sys.exit(1)
    print(f"All {len(results)} checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
