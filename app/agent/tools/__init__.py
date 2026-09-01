"""Tool surface for the text-path agent — factories bound to a request's DB
session + user_id (CONTEXT.md §4).

Six tools now: `remember`/`recall` (FR-5.3, FR-5.4) join the four from Phase
4 now that `app/memory/` exists — the "don't ship a stub" reasoning that kept
them out before no longer applies.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool

from app.agent.tools.logging import build_log_meal_tool, build_revise_meal_tool
from app.agent.tools.memory import build_recall_tool, build_remember_tool
from app.agent.tools.query import build_get_daily_totals_tool, build_search_meals_tool


def build_tools(session: Any, user_id: str) -> list[BaseTool]:
    return [
        build_log_meal_tool(session, user_id),
        build_revise_meal_tool(session, user_id),
        build_get_daily_totals_tool(session, user_id),
        build_search_meals_tool(session, user_id),
        build_remember_tool(session, user_id),
        build_recall_tool(session, user_id),
    ]
