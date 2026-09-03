"""`get_daily_totals` / `search_meals` — read-only tools over `mealops`/`repo`."""


from __future__ import annotations

from datetime import date as date_cls
from datetime import timedelta
from typing import Any

from langchain_core.tools import BaseTool, tool

from app.db.repo import recent_meals as _recent_meals
from app.mealops.logging_ops import get_totals as _get_totals


def _parse_date(value: str | None) -> date_cls:
    if not value or value.strip().lower() == "today":
        return date_cls.today()
    if value.strip().lower() == "yesterday":
        return date_cls.today() - timedelta(days=1)
    return date_cls.fromisoformat(value.strip())


def _fmt_totals(t: Any) -> str:
    return f"{t.kcal:.0f} kcal, {t.protein_g:.0f}g protein, {t.carbs_g:.0f}g carbs, {t.fat_g:.0f}g fat"


def build_get_daily_totals_tool(session: Any, user_id: str) -> BaseTool:
    @tool
    async def get_daily_totals(date: str | None = None) -> str:
        """Today's (or a given date's) logged meals and running macro totals."""
        day = _parse_date(date)
        result = await _get_totals(session, user_id, day=day)
        if not result.meals:
            return f"No meals logged for {day.isoformat()}."
        per_meal = "; ".join(
            f"{m.meal_slot}: {m.description or 'meal'} ({_fmt_totals(m.totals)})" for m in result.meals
        )
        return f"{day.isoformat()} totals: {_fmt_totals(result.totals)}. Meals — {per_meal}."

    return get_daily_totals


def build_search_meals_tool(session: Any, user_id: str) -> BaseTool:
    @tool
    async def search_meals(date: str | None = None, limit: int = 5) -> str:
        """Look up past logged meals by date (e.g. 'yesterday') to copy forward."""
        target = _parse_date(date) if date else None
        since = target or (date_cls.today() - timedelta(days=2))
        meals = await _recent_meals(session, user_id, since_date=since)
        if target is not None:
            meals = [m for m in meals if m.local_date == target]
        meals = meals[:limit]
        if not meals:
            label = target.isoformat() if target else "recently"
            return f"No meals found for {label}."
        lines = []
        for m in meals:
            items_txt = ", ".join(f"{i.name} x{i.quantity}{i.unit} (item_id={i.id})" for i in m.items)
            lines.append(f"{m.local_date.isoformat()} {m.meal_slot} (meal_id={m.id}): {items_txt}")
        return " | ".join(lines)

    return search_meals
