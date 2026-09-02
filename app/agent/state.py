"""Typed graph state for the text-path agent (FR-4.2).

`messages` uses LangGraph's `add_messages` reducer so each node returns only
the messages it adds, not the whole history. `last_meal_id` and
`pending_clarification` are the seams §5/§6 build on (memory-aware routines,
ambiguity tracking) — this phase writes `last_meal_id` (from `log_meal`/
`revise_meal` tool results, via `Command` updates) but doesn't yet branch on
`pending_clarification`; the one-question-per-turn rule is enforced by the
system prompt, not graph control flow, until there's a reason to make it
stateful.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    messages: Annotated[list[Any], add_messages]
    user_id: str
    last_meal_id: str | None
    pending_clarification: str | None
    image_id: str | None
    vision_observation: Any | None
    prefetch_data: Any | None
