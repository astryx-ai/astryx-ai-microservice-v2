from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Any, List, Tuple
import json

# LLM-based decision support
from app.services.llms.azure_openai import decision_model
from app.services.agent.state import AVAILABLE_ROUTES
from langchain_core.messages import SystemMessage, HumanMessage


def get_current_datetime_string() -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%d %H:%M UTC")


def _contains_any(text: str, needles: Iterable[str]) -> bool:
    lowered = (text or "").lower()
    return any(n in lowered for n in needles)


def needs_recency_injection(query: str) -> bool:
    keywords = [
        "today","now","current","latest","recent","this week","past week","as of",
        "up to date","uptodate","breaking","new","just released","update","updated",
        "real-time","realtime","live","this month","this quarter","this year","ytd",
    ]
    decision = _contains_any(query, keywords)
    if decision:
        print(f"[Helper] needs_recency_injection: True for query='{query}'")
    else:
        print(f"[Helper] needs_recency_injection: False for query='{query}'")
    return decision


def inject_datetime_into_query(query: str) -> str:
    injected = f"{query} (as of {get_current_datetime_string()})".strip()
    print(f"[Helper] inject_datetime_into_query: '{injected}'")
    return injected


def _summarize_context_for_router(messages: List[Any] | None) -> str:
    if not messages:
        return ""
    parts: List[str] = []
    for m in messages[-6:]:
        try:
            role = getattr(m, "type", "")
            content = (getattr(m, "content", None) or "")
            if not content:
                continue
            if role == "system":
                continue
            role_name = "User" if role == "human" else "Assistant"
            parts.append(f"{role_name}: {content[:400]}")
        except Exception:
            continue
    return "\n".join(parts[-4:])


def _llm_route_decision_multi(
    query: str,
    has_context: bool,
    context_summary: str,
    available_routes: List[str],
) -> Tuple[str, str]:
    try:
        model = decision_model(temperature=0.0)
        if hasattr(model, "streaming"):
            model.streaming = False
        sys = (
            "You are a routing controller. Your job is to pick exactly ONE route "
            "from the provided list of available subgraphs.\n"
            "Route Guidelines:\n"
            "- 'chart_viz': For requests asking to create charts, graphs, visualizations, or show data in visual format\n"
            "- 'deep_research': For comprehensive analysis requiring multi-step research and synthesis\n"
            "- 'standard': For quick searches, follow-ups, and general queries\n"
            "Pick the subgraph that best matches the task. "
            "Respond strictly as JSON with 'route' and 'reason'."
        )
        user = (
            f"Query: {query}\n"
            f"HasContext: {has_context}\n"
            f"ContextSummary:\n{context_summary or '(none)'}\n"
            f"AvailableRoutes: {available_routes}\n"
            "Return JSON: {\"route\": \"<one_of_available_routes>\", \"reason\": \"...\"}"
        )
        print("[Helper] Calling multi-route decision model (non-streaming)")
        resp = model.invoke([SystemMessage(content=sys), HumanMessage(content=user)])
        text = getattr(resp, "content", "") if hasattr(resp, "content") else str(resp)
        data = json.loads(text) if text.strip().startswith("{") else {}
        route = str(data.get("route") or "standard")
        reason = str(data.get("reason") or "")
        if route not in available_routes:
            print(f"[Helper] Invalid route '{route}', defaulting to 'standard'")
            route = "standard"
        print(f"[Helper] LLM chose route={route} | reason={reason}")
        return route, reason
    except Exception as e:
        print(f"[Helper] Multi-route LLM routing failed, fallback: {e}")
        return "standard", "routing model unavailable"


def decide_route(
    query: str,
    has_context: bool = False,
    context_messages: List[Any] | None = None,
    available_routes: List[str] | None = None,
) -> Tuple[str, str]:
    if available_routes is None:
        available_routes = AVAILABLE_ROUTES
    context_summary = _summarize_context_for_router(context_messages)
    route, reason = _llm_route_decision_multi(query, has_context, context_summary, available_routes)
    if route:
        return route, reason
    query_lower = query.lower()
    chart_keywords = [
        "chart", "graph", "visualization", "visualize", "plot", "bar chart",
        "pie chart", "line chart", "show data", "create chart", "display data",
        "visual", "dashboard", "infographic"
    ]
    if any(keyword in query_lower for keyword in chart_keywords):
        return "chart_viz", "explicit chart/visualization requested"
    if "deep research" in query_lower or "comprehensive analysis" in query_lower:
        return "deep_research", "explicit deep research requested"
    if len(query.split()) > 15:
        return "deep_research", "query too complex for standard route"
    return "standard", "default heuristic"


def requires_route(
    route_name: str,
    query: str,
    has_context: bool = False,
    context_messages: List[Any] | None = None,
    available_routes: List[str] | None = None,
) -> bool:
    chosen, reason = decide_route(query, has_context, context_messages, available_routes)
    print(f"[Helper] requires_route('{route_name}') → chosen={chosen} reason={reason}")
    return chosen == route_name


