"""Logging & totals engine — layer above repo.py."""


from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel

from app.db import repo
from app.db.models import Meal, MealItem
from app.nutrition.resolve import RawItem, ResolvedItem, resolve

ReviseAction = str  # "set_item_qty" | "remove_item" | "add_item" | "replace_items" | "delete_meal"


class MealLoggingError(Exception):
    """Domain error — e.g. `meal_ref="last"` with no active meal for the user."""


class ItemView(BaseModel):
    id: str
    name: str
    canonical_key: str | None
    quantity: float
    unit: str
    kcal: float
    protein_g: float
    carbs_g: float
    fat_g: float
    fiber_g: float | None
    nutrition_source: str
    confidence: float | None


class TotalsView(BaseModel):
    kcal: float
    protein_g: float
    carbs_g: float
    fat_g: float
    fiber_g: float


class MealView(BaseModel):
    meal_id: str
    meal_slot: str
    local_date: date
    status: str
    description: str | None
    items: list[ItemView]
    totals: TotalsView


class LogMealResult(BaseModel):
    meal: MealView
    daily_totals: TotalsView


class ReviseMealResult(BaseModel):
    meal: MealView
    daily_totals: TotalsView


class TotalsResult(BaseModel):
    day: date
    totals: TotalsView
    meals: list[MealView]


def _as_raw_items(items: list[RawItem] | list[dict[str, Any]]) -> list[RawItem]:
    return [i if isinstance(i, RawItem) else RawItem.model_validate(i) for i in items]


def _fmt_qty(q: float) -> str:
    return str(int(q)) if float(q).is_integer() else str(q)


def _describe(items: list[ResolvedItem]) -> str:
    return " + ".join(f"{_fmt_qty(r.quantity)} {r.name}" for r in items)


def _meal_confidence(items: list[ResolvedItem]) -> float:
    # Min, not mean — the meal-level confidence is only as good as its
    # weakest-resolved item, since that's the one worth hedging the reply on.
    return min((r.confidence for r in items), default=0.0)


def _item_payload(item: MealItem) -> dict[str, Any]:
    """MealItem -> plain dict of the columns `MealItem(**item)` accepts.

    Floats, not Decimal: SQLite's `Numeric` columns round-trip as `Decimal`,
    which `json.dumps` (used for the `meal_revisions.before/after` JSON
    columns) can't serialize.
    """
    return {
        "name": item.name,
        "canonical_key": item.canonical_key,
        "quantity": float(item.quantity),
        "unit": item.unit,
        "kcal": float(item.kcal),
        "protein_g": float(item.protein_g),
        "carbs_g": float(item.carbs_g),
        "fat_g": float(item.fat_g),
        "fiber_g": float(item.fiber_g) if item.fiber_g is not None else None,
        "nutrition_source": item.nutrition_source,
        "confidence": float(item.confidence) if item.confidence is not None else None,
    }


def _item_view(item: MealItem) -> ItemView:
    payload = _item_payload(item)
    return ItemView(id=item.id, **payload)


def _meal_view(meal: Meal) -> MealView:
    items = [_item_view(i) for i in meal.items]
    totals = TotalsView(
        kcal=sum(i.kcal for i in items),
        protein_g=sum(i.protein_g for i in items),
        carbs_g=sum(i.carbs_g for i in items),
        fat_g=sum(i.fat_g for i in items),
        fiber_g=sum(i.fiber_g or 0.0 for i in items),
    )
    return MealView(
        meal_id=meal.id,
        meal_slot=meal.meal_slot,
        local_date=meal.local_date,
        status=meal.status,
        description=meal.description,
        items=items,
        totals=totals,
    )


def _totals_view(totals: repo.Totals) -> TotalsView:
    return TotalsView(
        kcal=float(totals.kcal),
        protein_g=float(totals.protein_g),
        carbs_g=float(totals.carbs_g),
        fat_g=float(totals.fat_g),
        fiber_g=float(totals.fiber_g),
    )


async def log_meal(
    session,
    user_id: str,
    raw_items: list[RawItem] | list[dict[str, Any]],
    meal_slot: str,
    when: datetime,
    source: str,
    raw_input: str | None = None,
) -> LogMealResult:
    items = _as_raw_items(raw_items)
    if not items:
        raise MealLoggingError("log_meal called with no items")

    resolved = await resolve(items)
    # `when` is taken as already expressed in the user's local time (an
    # agent-layer concern once the graph has the user's `tz` in hand) — this
    # module denormalises local_date from it but does no tz lookup itself.
    local_date = when.date()

    meal = await repo.insert_meal(
        session,
        user_id=user_id,
        logged_at=when,
        local_date=local_date,
        meal_slot=meal_slot,
        source=source,
        items=[r.model_dump() for r in resolved],
        description=_describe(resolved),
        confidence=_meal_confidence(resolved),
        raw_input=raw_input,
    )

    daily = await repo.daily_totals(session, user_id, local_date)
    return LogMealResult(meal=_meal_view(meal), daily_totals=_totals_view(daily))


async def _resolve_meal_ref(session, user_id: str, meal_ref: str) -> Meal:
    if meal_ref == "last":
        meal = await repo.last_active_meal(session, user_id)
        if meal is None:
            raise MealLoggingError(f"no active meal found for user {user_id}")
        return meal

    meal = await repo.get_meal(session, meal_ref)
    if meal is None or meal.user_id != user_id or meal.status != "active":
        raise MealLoggingError(f"no active meal {meal_ref!r} for user {user_id}")
    return meal


async def _apply_action(action: str, meal: Meal, kwargs: dict[str, Any]) -> list[dict[str, Any]]:
    items = meal.items

    if action == "set_item_qty":
        item_id = kwargs["item_id"]
        quantity = kwargs["quantity"]
        target = next((i for i in items if i.id == item_id), None)
        if target is None:
            raise MealLoggingError(f"no item {item_id!r} on meal {meal.id}")
        [resolved] = await resolve([RawItem(name=target.name, quantity=quantity, unit=target.unit)])
        new_payload = resolved.model_dump()
        return [new_payload if i.id == item_id else _item_payload(i) for i in items]

    if action == "remove_item":
        item_id = kwargs["item_id"]
        remaining = [i for i in items if i.id != item_id]
        if len(remaining) == len(items):
            raise MealLoggingError(f"no item {item_id!r} on meal {meal.id}")
        return [_item_payload(i) for i in remaining]

    if action == "add_item":
        raw = RawItem(name=kwargs["name"], quantity=kwargs.get("quantity", 1.0), unit=kwargs.get("unit"))
        [resolved] = await resolve([raw])
        return [_item_payload(i) for i in items] + [resolved.model_dump()]

    if action == "replace_items":
        raw = _as_raw_items(kwargs["items"])
        if not raw:
            raise MealLoggingError("replace_items called with no items")
        resolved = await resolve(raw)
        return [r.model_dump() for r in resolved]

    raise MealLoggingError(f"unknown revise action {action!r}")


async def revise_meal(
    session,
    user_id: str,
    meal_ref: str,
    action: str,
    reason: str | None = None,
    **action_kwargs: Any,
) -> ReviseMealResult:
    meal = await _resolve_meal_ref(session, user_id, meal_ref)
    before = [_item_payload(i) for i in meal.items]

    if action == "delete_meal":
        await repo.soft_delete_meal(session, meal.id)
        after: list[dict[str, Any]] = []
    else:
        after = await _apply_action(action, meal, action_kwargs)
        # UPDATE of the existing meal row, never a new insert — the whole
        # correctness invariant this module exists to enforce.
        await repo.replace_meal_items(session, meal.id, after)

    await repo.insert_revision(session, meal.id, kind=action, before=before, after=after, reason=reason)

    meal_after = await repo.get_meal(session, meal.id)
    daily = await repo.daily_totals(session, user_id, meal_after.local_date)
    return ReviseMealResult(meal=_meal_view(meal_after), daily_totals=_totals_view(daily))


async def get_totals(session, user_id: str, day: date | None = None) -> TotalsResult:
    day = day or date.today()
    totals = await repo.daily_totals(session, user_id, day)
    meals = [m for m in await repo.recent_meals(session, user_id, day) if m.local_date == day]
    return TotalsResult(day=day, totals=_totals_view(totals), meals=[_meal_view(m) for m in meals])
