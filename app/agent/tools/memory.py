"""`remember` / `recall` tool factories."""


from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool, tool

from app.db.repo import get_active_memories
from app.memory.store import write_memory


def build_remember_tool(session: Any, user_id: str) -> BaseTool:
    @tool
    async def remember(kind: str, key: str, value: str) -> str:
        """Store an explicit durable fact the user stated (diet|goal|routine|alias|preference|dislike)."""
        await write_memory(
            session,
            user_id=user_id,
            kind=kind,
            key=key,
            value={"text": value},
            confidence=1.0,
            source_message=value,
        )
        return f"Got it, remembered: {key} = {value}."

    return remember


def build_recall_tool(session: Any, user_id: str) -> BaseTool:
    @tool
    async def recall(query: str) -> str:
        """Look up a stored fact not already in context, by keyword or kind."""
        memories = await get_active_memories(session, user_id)
        needle = query.strip().lower()
        hits = [
            m
            for m in memories
            if needle in m.kind.lower() or needle in m.key.lower() or needle in str(m.value).lower()
        ]
        if not hits:
            return f"No stored fact matches {query!r}."
        return " | ".join(f"{m.kind}/{m.key}: {m.value}" for m in hits)

    return recall
