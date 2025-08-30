from __future__ import annotations
from typing import Any, Dict, List, Optional
import json


def call_chart_local(
    *,
    query: str = "",
    symbol: str = "",
    chart_type: str = "",
    range_: str = "1d",
    interval: str = "5m",
    title: str = "",
    description: str = "",
    user_id: str = "",
    chat_id: str = "",
) -> Dict[str, Any]:
    """Return chart payload using local graph/builder (no gRPC).

    Matches the shape previously returned by call_chart_via_grpc.
    """
    from app.graph.runner import run_chart

    result = run_chart({
        "query": query,
        "symbol": symbol,
        "chart_type": chart_type,
        "range": range_ or "1d",
        "interval": interval or "5m",
        "title": title,
        "description": description,
        "user_id": user_id,
        "chat_id": chat_id,
    })
    try:
        return json.loads(result.get("json", "{}"))
    except Exception:
        return {}


def call_stock_local(*, ticker: Optional[str], exchange: Optional[str], company: Optional[str]) -> Dict[str, Any]:
    """Compute stock data locally using the existing stock node.

    Returns a dict compatible with AgentState['stock_data'].
    """
    from app.agents.super.state import AgentState
    from app.agents.super.stock import get_stock_node

    st: AgentState = {"ticker": ticker, "exchange": exchange, "company": company}
    st = get_stock_node(st)
    return st.get("stock_data") or {}


def call_news_local(*, ticker: Optional[str], company: Optional[str]) -> List[Dict[str, Any]]:
    """Compute news list locally using the existing news node.

    Returns a list of news item dicts.
    """
    from app.agents.super.state import AgentState
    from app.agents.super.news import get_news_node

    st: AgentState = {"ticker": ticker, "company": company}
    st = get_news_node(st)  # type: ignore
    return st.get("news_items") or []
