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


def _has_explicit_timeframe(query: str) -> bool:
    """Return True if the query already specifies a timeframe (years, months, quarters, FY, ranges)."""
    import re
    q = (query or "")
    ql = q.lower()
    # If already contains explicit 'as of', treat as explicit timeframe
    if "as of" in ql:
        return True
    # Year like 2019, 2020, 2024, 2025
    if re.search(r"\b(19|20)\d{2}\b", q):
        return True
    # Year ranges like 2020-2025
    if re.search(r"\b(19|20)\d{2}\s*[-–]\s*(19|20)\d{2}\b", q):
        return True
    # Fiscal year markers
    if "fy" in ql or "q1" in ql or "q2" in ql or "q3" in ql or "q4" in ql:
        return True
    # Month names
    months = [
        "january","february","march","april","may","june",
        "july","august","september","october","november","december",
    ]
    if any(m in ql for m in months):
        return True
    return False


def needs_recency_injection(query: str) -> bool:
    """Decide if we should inject current datetime.

    True when:
    - The query contains explicit recency keywords; OR
    - The query lacks any explicit timeframe (years/months/quarters/FY).
    """
    keywords = [
        "today","now","current","latest","recent","this week","past week","as of",
        "up to date","uptodate","breaking","new","just released","update","updated",
        "real-time","realtime","live","this month","this quarter","this year","ytd",
    ]
    if _contains_any(query, keywords):
        print(f"[Helper] needs_recency_injection: True (recency keywords) for query='{query}'")
        return True
    if not _has_explicit_timeframe(query):
        print(f"[Helper] needs_recency_injection: True (no explicit timeframe) for query='{query}'")
        return True
    print(f"[Helper] needs_recency_injection: False (explicit timeframe present) for query='{query}'")
    return False


def inject_datetime_into_query(query: str) -> str:
    """Append current datetime if not already present. Idempotent."""
    q = query or ""
    ql = q.lower()
    if "as of" in ql:
        # Already injected or provided by user
        print("[Helper] inject_datetime_into_query: skipped (already contains 'as of')")
        return q
    injected = f"{q} (as of {get_current_datetime_string()})".strip()
    print(f"[Helper] inject_datetime_into_query: '{injected}'")
    return injected


def sanitize_query_for_recency(query: str) -> str:
    """If intent is to get current data, strip stale explicit years from the query.

    - Removes standalone year tokens like 2019..2024 when recency intent is detected.
    - Leaves query unchanged when no recency intent.
    """
    import re
    if not needs_recency_injection(query):
        return query
    cleaned = re.sub(r"\b(19|20)\d{2}\b", "", query or "").strip()
    # Collapse extra spaces
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    if cleaned != query:
        print(f"[Helper] sanitize_query_for_recency: '{query}' -> '{cleaned}'")
    return cleaned


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
            "Route Guidelines (in priority order):\n"
            "1. 'financial_analysis': For corporate financial analysis, shareholding patterns, governance data, XBRL analysis, BSE/NSE analysis, ownership patterns, promoter holdings\n"
            "2. 'chart_viz': For requests EXPLICITLY asking to create charts, graphs, visualizations, or show data in visual format\n"
            "3. 'deep_research': For comprehensive analysis requiring multi-step research and synthesis\n"
            "4. 'standard': For quick searches, follow-ups, and general queries\n"
            "CRITICAL: ANY query about shareholding, ownership, promoters, corporate data, BSE/NSE must go to 'financial_analysis' - NOT chart_viz. "
            "Only choose 'chart_viz' if user explicitly requests visual charts/graphs. "
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
    
    # First, check keywords for financial analysis (most reliable)
    query_lower = query.lower()
    financial_keywords = [
        "shareholding", "shareholding pattern", "governance", "corporate governance", "board composition",
        "promoter holding", "promoter", "promoters", "institutional holding", "foreign holding", "xbrl",
        "director", "audit committee", "financial filing", "annual report",
        "bse filing", "compliance", "ownership pattern", "ownership", "stakeholders",
        "holding pattern", "fii", "dii", "bse code", "regulatory filings",
        "financial analysis", "corporate analysis", "price movement", "price analysis", 
        "stock performance", "news correlation", "price trend", "monthly high", "monthly low",
        "stock price", "price change", "volatility", "trading volume"
    ]
    
    # PRIORITY: Financial analysis has absolute priority
    if any(keyword in query_lower for keyword in financial_keywords):
        return "financial_analysis", "financial/corporate analysis requested (keyword priority)"
    
    # Then try LLM routing for other cases
    route, reason = _llm_route_decision_multi(query, has_context, context_summary, available_routes)
    if route:
        return route, reason
        
    # Fallback to other keyword checks
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


