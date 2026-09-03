"""`log_meal` / `revise_meal` tool factories — wraps over `mealops.logging_ops`."""


from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.types import Command
from pydantic import BaseModel

from app.mealops.logging_ops import (
    LogMealResult,
    MealLoggingError,
    ReviseMealResult,
)
from app.mealops.logging_ops import log_meal as _log_meal
from app.mealops.logging_ops import revise_meal as _revise_meal


class MealItemArg(BaseModel):
    name: str
    quantity: float = 1.0
    unit: str | None = None


def _fmt_totals(t: Any) -> str:
    return f"{t.kcal:.0f} kcal, {t.protein_g:.0f}g protein, {t.carbs_g:.0f}g carbs, {t.fat_g:.0f}g fat"


def _fmt_result(label: str, result: LogMealResult | ReviseMealResult) -> str:
    items_txt = "; ".join(
        f"{i.name} x{i.quantity}{i.unit} (item_id={i.id}, {i.kcal:.0f} kcal)" for i in result.meal.items
    )
    return (
        f"{label}: meal_id={result.meal.meal_id} [{result.meal.meal_slot}] {items_txt or '(no items)'}. "
        f"Meal total {_fmt_totals(result.meal.totals)}. "
        f"Daily totals so far: {_fmt_totals(result.daily_totals)}."
    )


def build_log_meal_tool(session: Any, user_id: str) -> BaseTool:
    @tool
    async def log_meal(
        *,
        items: list[MealItemArg],
        meal_slot: str,
        when: str | None = None,
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Log a new meal (items + meal_slot); returns per-item ids and fresh daily totals."""
        # `mealops.log_meal` takes `when` as already user-local time and derives
        # `local_date` from it directly (no tz lookup of its own — see its
        # docstring). Per-user tz isn't wired through the graph this phase, so
        # this defaults to host-local time, matching the host-local
        # `date.today()` that `get_daily_totals`/`search_meals` default to —
        # otherwise "today" could disagree across a UTC/local day boundary.
        ts = datetime.fromisoformat(when) if when else datetime.now().astimezone()
        raw_items = [i.model_dump() for i in items]
        try:
            result = await _log_meal(session, user_id, raw_items, meal_slot=meal_slot, when=ts, source="text")
        except MealLoggingError as exc:
            return Command(update={"messages": [ToolMessage(content=str(exc), tool_call_id=tool_call_id)]})
        return Command(
            update={
                "last_meal_id": result.meal.meal_id,
                "messages": [ToolMessage(content=_fmt_result("Logged", result), tool_call_id=tool_call_id)],
            }
        )

    return log_meal


def build_revise_meal_tool(session: Any, user_id: str) -> BaseTool:
    @tool
    async def revise_meal(
        *,
        action: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        meal_ref: str = "last",
        item_id: str | None = None,
        quantity: float | None = None,
        name: str | None = None,
        unit: str | None = None,
        items: list[MealItemArg] | None = None,
        reason: str | None = None,
    ) -> Command:
        """Fix an already-logged meal: set_item_qty|remove_item|add_item|replace_items|delete_meal."""
        kwargs: dict[str, Any] = {}
        if item_id is not None:
            kwargs["item_id"] = item_id
        if quantity is not None:
            kwargs["quantity"] = quantity
        if name is not None:
            kwargs["name"] = name
        if unit is not None:
            kwargs["unit"] = unit
        if items is not None:
            kwargs["items"] = [i.model_dump() for i in items]
        try:
            result = await _revise_meal(session, user_id, meal_ref=meal_ref, action=action, reason=reason, **kwargs)
        except MealLoggingError as exc:
            return Command(update={"messages": [ToolMessage(content=str(exc), tool_call_id=tool_call_id)]})
        return Command(
            update={
                "last_meal_id": result.meal.meal_id,
                "messages": [ToolMessage(content=_fmt_result("Revised", result), tool_call_id=tool_call_id)],
            }
        )

    return revise_meal
