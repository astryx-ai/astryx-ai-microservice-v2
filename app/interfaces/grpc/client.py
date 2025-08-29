from __future__ import annotations
from typing import Any, Dict, List, Optional
import os
import json
import asyncio


def _grpc_stub():
    """Return (channel, stub) if gRPC modules are available, else (None, None).

    Uses env GRPC_SERVER_ADDR like "localhost:50051".
    """
    try:
        import grpc  # type: ignore
        # Ensure generated modules are importable (as in server)
        import sys
        import pathlib
        proto_dir = pathlib.Path(__file__).resolve().parents[2] / "proto"
        p = str(proto_dir)
        if p not in sys.path:
            sys.path.insert(0, p)
        import message_pb2  # type: ignore
        import message_pb2_grpc  # type: ignore
    except Exception:
        return None, None, None, None
    addr = os.getenv("GRPC_SERVER_ADDR", f"localhost:{os.getenv('GRPC_PORT','50051')}")
    channel = grpc.insecure_channel(addr)  # type: ignore
    stub = message_pb2_grpc.MessageServiceStub(channel)  # type: ignore
    return channel, stub, message_pb2, grpc


def _yf_symbol(ticker: str, exchange: Optional[str]) -> str:
    if not ticker:
        return ""
    if exchange == "NSE":
        return f"{ticker}.NS"
    if exchange == "BSE":
        return f"{ticker}.BO"
    return ticker


def call_chart_via_grpc(
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
    """Call GetChart over gRPC and return parsed payload dict.

    Falls back to local builder if gRPC is unavailable.
    """
    channel, stub, message_pb2, grpc_mod = _grpc_stub()
    if not stub or not message_pb2:
        # Fallback to local builder
        from app.features.charts.builder import build_chart  # type: ignore
        try:
            payload = asyncio.run(build_chart(
                query=query,
                symbol=symbol,
                chart_type=chart_type,
                range_=range_ or "1d",
                interval=interval or "5m",
                title=title,
                description=description,
            ))
        except RuntimeError:
            loop = asyncio.get_event_loop()
            payload = loop.run_until_complete(build_chart(
                query=query,
                symbol=symbol,
                chart_type=chart_type,
                range_=range_ or "1d",
                interval=interval or "5m",
                title=title,
                description=description,
            ))  # type: ignore
        return payload if isinstance(payload, dict) else json.loads(payload or "{}")
    req = message_pb2.ChartRequest(
        query=query,
        symbol=symbol,
        chart_type=chart_type,
        range=range_,
        interval=interval,
        title=title,
        description=description,
        user_id=user_id,
        chat_id=chat_id,
    )
    resp = stub.GetChart(req)
    try:
        return json.loads(getattr(resp, "json", "") or "{}")
    except Exception:
        return {}


def call_stock_via_grpc_or_local(
    *,
    ticker: Optional[str],
    exchange: Optional[str],
    company: Optional[str],
) -> Dict[str, Any]:
    """Try to call a gRPC GetStock method if available; otherwise compute locally.

    Returns a dict compatible with AgentState['stock_data'].
    """
    # Try gRPC dynamic method if provided by server
    channel, stub, message_pb2, grpc_mod = _grpc_stub()
    if stub and message_pb2:
        get_stock = getattr(stub, "GetStock", None)
        if callable(get_stock):
            # If proto exists, it likely has fields ticker/exchange/company
            try:
                StockRequest = getattr(message_pb2, "StockRequest")
                req = StockRequest(
                    ticker=ticker or "",
                    exchange=exchange or "",
                    company=company or "",
                )
                resp = get_stock(req)
                payload_json = getattr(resp, "json", "")
                return json.loads(payload_json or "{}")
            except Exception:
                pass
    # Fallback to local node logic
    from app.agents.super.state import AgentState  # type: ignore
    from app.agents.super.stock import get_stock_node  # type: ignore
    st: AgentState = {"ticker": ticker, "exchange": exchange, "company": company}
    st = get_stock_node(st)
    return st.get("stock_data") or {}


def call_news_via_grpc_or_local(
    *,
    ticker: Optional[str],
    company: Optional[str],
) -> List[Dict[str, Any]]:
    """Try to call a gRPC GetNews method if available; otherwise compute locally.

    Returns a list of news item dicts.
    """
    channel, stub, message_pb2, grpc_mod = _grpc_stub()
    if stub and message_pb2:
        get_news = getattr(stub, "GetNews", None)
        if callable(get_news):
            try:
                NewsRequest = getattr(message_pb2, "NewsRequest")
                req = NewsRequest(
                    ticker=ticker or "",
                    company=company or "",
                )
                resp = get_news(req)
                payload_json = getattr(resp, "json", "")
                data = json.loads(payload_json or "{}")
                return data.get("items") or []
            except Exception:
                pass
    # Fallback to local
    from app.agents.super.state import AgentState  # type: ignore
    from app.agents.super.news import get_news_node  # type: ignore
    st: AgentState = {"ticker": ticker, "company": company}
    st = get_news_node(st)  # type: ignore
    return st.get("news_items") or []
