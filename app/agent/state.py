"""Typed graph state for the text-path agent."""


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
    # Rendered string, not the Prefetch dataclass: graph state is checkpointed
    # via msgpack, which can't serialize ORM rows.
    prefetch_block: str | None
