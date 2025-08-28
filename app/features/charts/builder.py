from __future__ import annotations
import re
import json
from typing import Any, Dict, List, Optional, Tuple
import httpx
from .chart import SUPPORTED_CHART_FORMATS  # type: ignore

# Public contract
# build_chart(query: str = "", symbol: str = "", chart_type: str = "", range_: str = "1d", interval: str = "5m", title: str = "", description: str = "") -> Dict[str, Any]

_BLOCKED_SYMBOL_WORDS = {
    "candle", "candlestick", "ohcl", "ohlc", "chart", "line", "area", "bar",
    "stock", "stocks", "price", "prices", "for", "on", "with", "using",
    "nse", "bse", "india", "indian", "market", "markets"
}

_INDEX_MAP = [
    (re.compile(r"\b(nifty\s*50|nifty50|\bnifty\b)\b", re.I), "^NSEI"),
    (re.compile(r"\b(bank\s*nifty|nifty\s*bank)\b", re.I), "^NSEBANK"),
    (re.compile(r"\bsensex\b", re.I), "^BSESN"),
    (re.compile(r"\b(nifty\s*next\s*50|niftynext50)\b", re.I), "^NSMIDCP"),
]


def _infer_exchange(query: str) -> Optional[str]:
    q = (query or "").lower()
    if re.search(r"\bnse\b", q):
        return "NS"
    if re.search(r"\bbse\b", q):
        return "BO"
    return None


async def _yahoo_search_symbol(query: str, preferred_exch: Optional[str] = None) -> Optional[str]:
    q = (query or "").strip()
    if not q:
        return None
    base = "https://query2.finance.yahoo.com/v1/finance/search"
    params = {"q": q, "quotesCount": 10, "newsCount": 0}
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.yahoo.com/"}
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.get(base, params=params, headers=headers)
            r.raise_for_status()
            quotes = r.json().get("quotes", [])
            if preferred_exch in ("NS", "BO"):
                for q in quotes:
                    sym = q.get("symbol") or ""
                    if sym.endswith(f".{preferred_exch}"):
                        return sym
            for exc in ("NS", "BO"):
                for q in quotes:
                    sym = q.get("symbol") or ""
                    if sym.endswith(f".{exc}"):
                        return sym
            if quotes:
                return quotes[0].get("symbol")
    except Exception:
        return None
    return None


def _resolve_index_symbol(query: str) -> Optional[str]:
    for rx, sym in _INDEX_MAP:
        if rx.search(query or ""):
            return sym
    return None


def _clean_phrase(phrase: str) -> str:
    phrase = re.split(r"\s+(?:on|in|with|using|chart|line|area|bar|candle|candlestick)\b", phrase, flags=re.I)[0]
    return phrase.strip().strip("'\"")


async def _extract_symbol(query: str, symbol: str) -> str:
    sym = (symbol or "").strip()
    if sym:
        return sym
    idx = _resolve_index_symbol(query.lower())
    if idx:
        return idx
    pref_ex = _infer_exchange(query)
    m_for = re.search(r"\bfor\s+([A-Za-z0-9&\.\-\s]{2,40})", query, re.I)
    if m_for:
        phrase = _clean_phrase(m_for.group(1))
        if phrase:
            resolved = await _yahoo_search_symbol(phrase, pref_ex)
            if resolved:
                return resolved
    quotes = re.findall(r"[\"']([^\"']+)[\"']", query)
    if quotes:
        phrase = quotes[-1]
        resolved = await _yahoo_search_symbol(phrase, pref_ex)
        if resolved:
            return resolved
    tokens = re.findall(r"\b([A-Za-z0-9]{2,12}(?:\.(?:NS|BO))?)\b", query)
    for t in tokens[::-1]:
        base = t.split(".")[0]
        if base.lower() not in _BLOCKED_SYMBOL_WORDS:
            m = re.match(r"^([A-Za-z0-9]{1,12})(?:\.(NS|BO))?$", t)
            if m:
                tick = m.group(1).upper()
                suf = m.group(2)
                return f"{tick}.{suf}" if suf else tick
    resolved = await _yahoo_search_symbol(query, pref_ex)
    if resolved:
        return resolved
    return sym


async def _fetch_chart(symb: str, rng: str, intr: str) -> Optional[Dict[str, Any]]:
    base = f"https://query2.finance.yahoo.com/v8/finance/chart/{symb}"
    params = {"interval": intr, "range": rng, "includePrePost": "true", "events": "div|split|earn"}
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.yahoo.com/"}
    try:
        async with httpx.AsyncClient(timeout=12) as client:
            r = await client.get(base, params=params, headers=headers)
            r.raise_for_status()
            return r.json()["chart"]["result"][0]
    except Exception:
        return None


async def build_chart(*, query: str = "", symbol: str = "", chart_type: str = "", range_: str = "1d", interval: str = "5m", title: str = "", description: str = "") -> Dict[str, Any]:
    # Resolve symbol
    sym = await _extract_symbol(query, symbol)
    if sym and not sym.startswith("^") and not re.search(r"\.(NS|BO)$", sym):
        pref_ex = _infer_exchange(query)
        suffix = f".{pref_ex}" if pref_ex in ("NS", "BO") else ".NS"
        sym = sym + suffix
    # Infer chart type
    preferred_type = (chart_type or "").strip().lower()
    if not preferred_type and query:
        if re.search(r"\b(candle|candlestick|ohlc)\b", query, re.I):
            preferred_type = "candlestick-standard"
        elif re.search(r"\b(line)\b", query, re.I):
            preferred_type = "line-standard"
        elif re.search(r"\b(area)\b", query, re.I):
            preferred_type = "area-standard"
        elif re.search(r"\b(bar)\b", query, re.I):
            preferred_type = "bar-standard"
    if not preferred_type:
        preferred_type = "line-standard"
    # Fetch with fallbacks
    attempts: List[Tuple[str, str]] = [
        (range_ or "1d", interval or "5m"),
        ("5d", "15m"),
        ("1mo", "1d"),
        ("6mo", "1wk"),
        ("5y", "1mo"),
    ]
    chart = None
    for rng, intr in attempts:
        chart = await _fetch_chart(sym, rng, intr)
        if chart and chart.get("timestamp"):
            range_, interval = rng, intr
            break
    if not chart:
        return {"error": "Failed to fetch Yahoo chart after retries", "symbol": sym}
    # Build data rows
    timestamps: List[int] = chart.get("timestamp") or []
    inds = chart.get("indicators", {})
    quote = (inds.get("quote") or [{}])[0]
    close = quote.get("close") or []
    open_ = quote.get("open") or []
    high = quote.get("high") or []
    low = quote.get("low") or []
    volume = quote.get("volume") or []
    meta = chart.get("meta", {})
    name = meta.get("symbol") or sym
    from datetime import datetime, timezone
    def iso(ts: Optional[int]) -> Optional[str]:
        if not ts:
            return None
        try:
            return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        except Exception:
            return None
    data: List[Dict[str, Any]] = []
    for i, ts in enumerate(timestamps):
        row: Dict[str, Any] = {"time": iso(ts)}
        if i < len(close):
            row["close"] = close[i]
        if i < len(open_):
            row["open"] = open_[i]
        if i < len(high):
            row["high"] = high[i]
        if i < len(low):
            row["low"] = low[i]
        if i < len(volume):
            row["volume"] = volume[i]
        if any(row.get(k) is not None for k in ("open", "high", "low", "close")):
            data.append(row)
    ttl = title or f"{name} ({range_} / {interval})"
    desc = description or "Yahoo Finance price data"
    ct = preferred_type
    # Map to schema
    if ct == "candlestick-standard":
        chart_payload = {
            "type": "candlestick-standard",
            "title": ttl,
            "description": desc,
            "xAxisKey": "time",
            "groupedKeys": ["open", "high", "low", "close"],
            "nameKey": "time",
            "data": data,
        }
    elif ct.startswith("line"):
        chart_payload = {
            "type": "line-standard",
            "title": ttl,
            "description": desc,
            "dataKey": "close",
            "nameKey": "time",
            "color": "hsl(220, 75%, 55%)",
            "data": data,
        }
    elif ct.startswith("area"):
        chart_payload = {
            "type": "area-standard",
            "title": ttl,
            "description": desc,
            "dataKey": "close",
            "nameKey": "time",
            "color": "hsl(207, 90%, 54%)",
            "data": data,
        }
    elif ct.startswith("bar"):
        chart_payload = {
            "type": "bar-standard",
            "title": ttl,
            "description": desc,
            "dataKey": "volume",
            "nameKey": "time",
            "data": data,
        }
    else:
        chart_payload = {
            "type": "line-standard",
            "title": ttl,
            "description": desc,
            "dataKey": "close",
            "nameKey": "time",
            "color": "hsl(220, 75%, 55%)",
            "data": data,
        }
    # Optional: basic validation against schema keys
    try:
        spec = SUPPORTED_CHART_FORMATS.get(chart_payload.get("type", ""), {})
        required = spec.get("required_keys", [])
        missing = [k for k in required if k not in chart_payload]
        if missing:
            chart_payload["warning"] = f"Missing required keys: {', '.join(missing)}"
    except Exception:
        pass
    return chart_payload
