"""LangGraph text-path agent core (FR-4.1, FR-4.4-FR-4.7).

Graph shape follows CONTEXT.md §5's diagram, sized to what exists this phase:
`load_session` -> `agent` (tool loop) -> respond. The `memory_retrieve` /
`vision_extract` fan-out isn't built yet (memory is Phase 5, vision Phase 7,
neither exists in this repo), so `load_session` (done in `app/cli.py` before
the graph even runs — opening the session and resolving `user_id`) feeds
straight into `agent`. Adding those nodes later is wiring a parallel branch in
front of `agent`, not a rewrite — the seam is deliberate.

Session/user_id binding: tools need a live DB session and a user_id, but a
compiled LangGraph graph is normally built once and reused. Here `build_graph`
is called fresh per turn (see `app/cli.py`), closing the tools over that
turn's `(session, user_id)` — simple, correct, and cheap enough at this scale;
the checkpointer (passed in, not rebuilt) is what actually persists state
across turns, not the graph object.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.agent.prompts import SYSTEM_PROMPT
from app.agent.state import AgentState
from app.agent.tools import build_tools
from app.config import get_settings


def _trim_history(messages: list[BaseMessage], history_turns: int) -> list[BaseMessage]:
    """Keep only the last `history_turns` user turns (FR-4.7).

    Cuts only at HumanMessage boundaries so a tool-call/tool-response pair is
    never split apart — a HumanMessage never sits inside one of those.
    """
    human_seen = 0
    cut = 0
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            human_seen += 1
            if human_seen > history_turns:
                cut = i + 1
                break
    return messages[cut:]


def build_llm() -> BaseChatModel:
    settings = get_settings()
    return ChatOpenAI(
        model=settings.text_model,
        base_url=settings.llm_base_url,
        api_key=settings.require_api_key(),
        max_tokens=settings.reply_max_tokens,
    )


def build_graph(
    session: Any,
    user_id: str,
    checkpointer: BaseCheckpointSaver,
    llm: BaseChatModel | None = None,
) -> CompiledStateGraph:
    """Compile a fresh graph bound to this turn's session/user_id.

    `llm` is overridable so tests/verify scripts can swap in a scripted fake
    without touching provider config.
    """
    settings = get_settings()
    tools = build_tools(session, user_id)
    llm_with_tools = (llm or build_llm()).bind_tools(tools)

    async def agent_node(state: AgentState) -> dict:
        history = _trim_history(state["messages"], settings.history_turns)
        prompt = [SystemMessage(content=SYSTEM_PROMPT), *history]
        response = await llm_with_tools.ainvoke(prompt)
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=checkpointer)
