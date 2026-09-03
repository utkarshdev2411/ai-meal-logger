import json
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.agent.prompts import build_system_prompt
from app.agent.state import AgentState
from app.agent.tools import build_tools
from app.config import get_settings, next_api_key
from app.agent.prefetch import prefetch, render_prefetch_block
from app.vision.process import process_image_by_id
from app.telemetry import TurnTimer

def _trim_history(messages: list[BaseMessage], history_turns: int) -> list[BaseMessage]:
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
    if settings.text_model.startswith("gemini"):
        return ChatGoogleGenerativeAI(
            model=settings.text_model,
            google_api_key=next_api_key(),
            max_output_tokens=settings.llm_max_output_tokens,
        )
    return ChatOpenAI(
        model=settings.text_model,
        base_url=settings.llm_base_url,
        api_key=next_api_key(),
        max_tokens=settings.llm_max_output_tokens,
    )

def build_graph(
    session: Any,
    user_id: str,
    checkpointer: BaseCheckpointSaver,
    llm: BaseChatModel | None = None,
    timer: TurnTimer | None = None,
) -> CompiledStateGraph:
    settings = get_settings()
    tools = build_tools(session, user_id)
    llm_with_tools = (llm or build_llm()).bind_tools(tools)
    tool_node = ToolNode(tools)
    llm_calls = {"count": 0}

    async def prefetch_node(state: AgentState) -> dict:
        if timer:
            async with timer.phase("prefetch"):
                data = await prefetch(user_id)
        else:
            data = await prefetch(user_id)
        return {"prefetch_block": render_prefetch_block(data)}

    async def vision_node(state: AgentState) -> dict:
        image_id = state.get("image_id")
        if not image_id:
            return {"vision_observation": None}
        if timer:
            async with timer.phase("vision"):
                obs = await process_image_by_id(session, image_id)
        else:
            obs = await process_image_by_id(session, image_id)
        return {"vision_observation": obs}

    async def agent_node(state: AgentState) -> dict:
        vision_obs = state.get("vision_observation")
        sys_prompt = build_system_prompt(state.get("prefetch_block") or "")

        if isinstance(vision_obs, dict) and vision_obs.get("error"):
            # Degrade in words, not by pasting an error payload in as an observation.
            sys_prompt += (
                "\n\nVISION_NOTE: the photo could not be read. Ask the user to "
                "describe the plate instead; do not guess its contents."
            )
        elif vision_obs:
            sys_prompt += f"\n\nVISION_OBSERVATION:\n{json.dumps(vision_obs, indent=2)}"

        history = _trim_history(state["messages"], settings.history_turns)
        prompt = [SystemMessage(content=sys_prompt), *history]

        llm_calls["count"] += 1
        phase_name = "llm_1" if llm_calls["count"] == 1 else "llm_2"
        if timer:
            async with timer.phase(phase_name):
                response = await llm_with_tools.ainvoke(prompt)
        else:
            response = await llm_with_tools.ainvoke(prompt)
        return {"messages": [response]}

    async def tools_node(state: AgentState) -> dict:
        if timer:
            async with timer.phase("tool"):
                return await tool_node.ainvoke(state)
        return await tool_node.ainvoke(state)

    graph = StateGraph(AgentState)

    # We use a dummy start node to branch out
    async def start_node(state: AgentState) -> dict:
        return {}

    graph.add_node("start", start_node)
    graph.add_node("prefetch", prefetch_node)
    graph.add_node("vision", vision_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)

    graph.set_entry_point("start")
    
    # Conditional routing from start to allow parallel execution if image exists
    def route_start(state: AgentState) -> list[str]:
        if state.get("image_id"):
            return ["prefetch", "vision"]
        return ["prefetch"]
        
    graph.add_conditional_edges("start", route_start, ["prefetch", "vision"])
    
    graph.add_edge("prefetch", "agent")
    graph.add_edge("vision", "agent")
    
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=checkpointer)
