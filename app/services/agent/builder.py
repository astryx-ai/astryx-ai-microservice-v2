from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, SystemMessage

from app.services.llms.azure_openai import chat_model
from app.agent_tools.registry import load_tools
from app.agent_tools.helper_tools import decide_route
from app.utils.stream_utils import emit_process
from app.subgraphs.deep_research import run_deep_research
from app.subgraphs.chart_viz import run_chart_viz
from app.services.agent.state import AVAILABLE_ROUTES


def _route_decision(state):
    """Decide which subgraph to route to and persist route into the state."""
    messages = state.get("messages", [])
    if not messages:
        print("[Router] No messages, routing to standard")
        state["route"] = "standard"
        state["decision_reason"] = "No messages"
        return "standard"

    user_and_ai_messages = [
        msg for msg in messages if hasattr(msg, "type") and msg.type in ["human", "ai"]
    ]
    has_context = len(user_and_ai_messages) > 2

    # Extract user query (last human message)
    user_query = ""
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            content = msg.content
            user_query = content[6:] if content.startswith("Task: ") else content
            break

    # Cached decision check
    if state.get("route") and state.get("query") == user_query:
        cached_route = state["route"]
        print(f"[Router] Using cached route: {cached_route}")
        return cached_route

    try:
        available_routes = AVAILABLE_ROUTES
        chosen_route, reason = decide_route(
            user_query,
            has_context,
            context_messages=user_and_ai_messages,
            available_routes=available_routes,
        )
        print(f"[Router] Decision successful: {chosen_route}")
    except Exception as route_error:
        print(f"[Router] Decision failed: {route_error}")
        chosen_route, reason = "standard", f"routing error: {route_error}"

    # Persist decision in state
    state["route"] = chosen_route
    state["decision_reason"] = reason
    state["query"] = user_query
    state["context"] = user_and_ai_messages

    # Optional: emit process notification
    if chosen_route != "standard":
        try:
            emit_process({"message": f"Invoking {chosen_route.replace('_', ' ')}"})
        except Exception:
            pass

    print(f"[Router] Routing to {chosen_route} | reason={reason}")
    return chosen_route


def _standard_agent_node(state):
    """Standard agent node for simple queries."""
    print("[StandardAgent] Processing with standard search tools")
    messages = state.get("messages", [])

    # Last human message
    user_question = ""
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            content = msg.content
            user_question = content[6:] if content.startswith("Task: ") else content
            break

    # Context awareness check
    context_messages = [m for m in messages if hasattr(m, "type") and m.type in ["human", "ai"]]
    has_context = len(context_messages) > 2

    if has_context and any(
        ref in user_question.lower()
        for ref in ["you mentioned", "companies", "those", "these", "them", "comparison"]
    ):
        print("[StandardAgent] Using context-aware mode")
        enhanced_system = (
            "You are a financial AI assistant with access to conversation history. "
            "Focus on previously discussed topics for follow-ups and comparisons. "
            "Use web search only if truly necessary for fresh data."
        )
        updated_messages = [
            SystemMessage(content=enhanced_system) if getattr(msg, "type", None) == "system" else msg
            for msg in messages
        ]
        state = {"messages": updated_messages}

    llm = chat_model(temperature=0.2)
    tools = load_tools(use_cases=["web_search"], structured=True)
    agent = create_react_agent(llm, tools)
    return agent.invoke(state)


def _deep_research_node(state):
    """Deep research node for comprehensive analysis."""
    print("[DeepResearch] Processing with deep research subgraph")
    messages = state.get("messages", [])

    # Extract user question
    user_question = ""
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            content = msg.content
            user_question = content[6:] if content.startswith("Task: ") else content
            break

    print(f"[DeepResearch] Extracted user question: '{user_question}'")

    try:
        context_messages = [m for m in messages if hasattr(m, "type") and m.type in ["human", "ai"]]
        research_result = run_deep_research(user_question, context_messages=context_messages)

        if not research_result or not isinstance(research_result, str):
            research_result = (
                "I encountered an issue while conducting the deep research. Please try again."
            )

        return {"messages": messages + [AIMessage(content=research_result)]}

    except Exception as e:
        return {"messages": messages + [AIMessage(content=f"Deep research error: {str(e)}")]}


def _chart_viz_node(state):
    """Chart visualization node for creating data visualizations."""
    print("[ChartViz] Processing with chart visualization subgraph")
    messages = state.get("messages", [])

    # Extract user question
    user_question = ""
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            content = msg.content
            user_question = content[6:] if content.startswith("Task: ") else content
            break

    print(f"[ChartViz] Extracted user question: '{user_question}'")

    try:
        context_messages = [m for m in messages if hasattr(m, "type") and m.type in ["human", "ai"]]
        chart_result = run_chart_viz(user_question, context_messages=context_messages)

        if not chart_result or not isinstance(chart_result, str):
            chart_result = (
                "I encountered an issue while creating the chart visualization. Please try again."
            )

        return {"messages": messages + [AIMessage(content=chart_result)]}

    except Exception as e:
        return {"messages": messages + [AIMessage(content=f"Chart visualization error: {str(e)}")]}


def build_routed_agent():
    """Build a LangGraph with routing between multiple subgraphs."""
    graph = StateGraph(dict)

    # Nodes
    graph.add_node("standard", _standard_agent_node)
    graph.add_node("deep_research", _deep_research_node)
    graph.add_node("chart_viz", _chart_viz_node)
    # Future: graph.add_node("legal_analysis", _legal_analysis_node)

    # Router → nodes
    graph.add_conditional_edges(
        "__start__",
        _route_decision,
        {
            "standard": "standard",
            "deep_research": "deep_research",
            "chart_viz": "chart_viz",
        },
    )

    # Endpoints
    graph.add_edge("standard", END)
    graph.add_edge("deep_research", END)
    graph.add_edge("chart_viz", END)

    return graph.compile()


def build_agent(use_cases: list[str] | None = None, structured: bool = False):
    """Legacy compatibility shim."""
    return build_routed_agent()
