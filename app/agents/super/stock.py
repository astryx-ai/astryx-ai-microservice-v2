from __future__ import annotations
from typing import Any, Dict, Optional
from datetime import datetime, timezone
import asyncio

from .state import AgentState
from .utils import yf_symbol, yahoo_search_symbol, yahoo_chart, yahoo_quote, last_non_null


def _compute_snapshot_from_chart(chart: Dict[str, Any]) -> Dict[str, Any]:
    if not chart:
        return {}
    meta = chart.get("meta", {})
    q0 = chart.get("indicators", {}).get("quote", [{}])[0]
    # Close/High/Low/Volume from Yahoo can include None entries; filter them out safely
    closes = [float(v) for v in (q0.get("close") or []) if v is not None]
    highs = [float(v) for v in (q0.get("high") or []) if v is not None]
    lows = [float(v) for v in (q0.get("low") or []) if v is not None]
    vols = [float(v) for v in (q0.get("volume") or []) if v is not None]

    last_price = (closes[-1] if closes else None) or meta.get("regularMarketPrice")
    day_high = (max(highs) if highs else None) or meta.get("regularMarketDayHigh")
    day_low = (min(lows) if lows else None) or meta.get("regularMarketDayLow")
    volume = (int(sum(vols)) if vols else None) or meta.get("regularMarketVolume")

    prev_close = (
        meta.get("chartPreviousClose")
        or meta.get("previousClose")
        or meta.get("regularMarketPreviousClose")
    )
    # Compute percent change only when both numbers are present and previous close is non-zero
    if last_price is not None and prev_close not in (None, 0, 0.0):
        try:
            percent_change = (float(last_price) - float(prev_close)) / float(prev_close) * 100.0
        except Exception:
            percent_change = None
    else:
        percent_change = None
    market_cap = meta.get("marketCap")
    return {
        "current_price": last_price,
        "percent_change": percent_change,
        "daily_high": day_high,
        "daily_low": day_low,
        "market_cap": market_cap,
        "volume": volume,
    }


def get_stock_node(state: AgentState) -> AgentState:
    ticker = state.get("ticker")
    ex = state.get("exchange")
    company = state.get("company")

    if not ticker and company:
        y = yahoo_search_symbol(company)
        if y and y.get("ticker"):
            state["ticker"] = ticker = y["ticker"]
            state["exchange"] = ex = y.get("exchange")
            state["company"] = y.get("company") or company

    if ticker and not ex:
        y = yahoo_search_symbol(ticker)
        if y and y.get("ticker"):
            state["ticker"] = ticker = y["ticker"]
            state["exchange"] = ex = y.get("exchange")

    if not ticker:
        state["stock_data"] = None
        return state

    symbol = yf_symbol(ticker, ex)
    attempts = [("1m", "1d"), ("5m", "5d"), ("15m", "5d"), ("1d", "1mo")]
    snap: Dict[str, Any] = {}
    for interval, range_ in attempts:
        chart = yahoo_chart(symbol, interval=interval, range_=range_)
        if chart:
            snap = _compute_snapshot_from_chart(chart)
            break

    if not snap:
        q = yahoo_quote(symbol)
        if q:
            last_price = q.get("regularMarketPrice") or q.get("postMarketPrice") or q.get("preMarketPrice")
            prev_close = q.get("regularMarketPreviousClose") or q.get("previousClose")
            percent_change = ((last_price - prev_close) / prev_close * 100) if last_price and prev_close else None
            snap = {
                "current_price": last_price,
                "percent_change": percent_change,
                "daily_high": q.get("regularMarketDayHigh"),
                "daily_low": q.get("regularMarketDayLow"),
                "market_cap": q.get("marketCap"),
                "volume": q.get("regularMarketVolume"),
            }

    stock_payload = {
        "symbol": symbol,
        "exchange": ex,
        **snap,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    state["stock_data"] = stock_payload

    async def _ingest_snapshot(payload: Dict[str, Any]):
        try:
            # Lazy import to avoid hard dependency at import time
            try:
                from app.tools.rag import chunk_text, upsert_stocks  # type: ignore
            except Exception:
                chunk_text = upsert_stocks = None  # type: ignore
            if upsert_stocks and chunk_text:
                summary = f"{state.get('company') or ticker} stock: price={payload.get('current_price')}, pct={payload.get('percent_change')}, etc."
                meta = {"ticker": ticker, "company": state.get("company"), "type": "stock", "ts": payload["ts"]}
                upsert_stocks(chunk_text(summary, meta))
        except Exception:
            pass

    try:
        asyncio.get_running_loop().create_task(_ingest_snapshot(stock_payload))
    except RuntimeError:
        pass

    return state
