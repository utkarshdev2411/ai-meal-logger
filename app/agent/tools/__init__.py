"""Tool surface for the text-path agent — factories bound to a request's DB
session + user_id (CONTEXT.md §4).

Four tools this phase, not six: `remember`/`recall` are deliberately not
registered. `app/memory/` doesn't exist yet (Phase 5), and a no-op/stub tool
that looks real risked shipping silently broken — better to not offer it at
all and have the agent acknowledge durable statements ("i'm vegetarian btw")
in natural language, per `app/agent/prompts.py`.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool

from app.agent.tools.logging import build_log_meal_tool, build_revise_meal_tool
from app.agent.tools.query import build_get_daily_totals_tool, build_search_meals_tool


def build_tools(session: Any, user_id: str) -> list[BaseTool]:
    return [
        build_log_meal_tool(session, user_id),
        build_revise_meal_tool(session, user_id),
        build_get_daily_totals_tool(session, user_id),
        build_search_meals_tool(session, user_id),
    ]
