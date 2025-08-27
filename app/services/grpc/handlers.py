import json, re, os, asyncio
from typing import Dict, Any, Optional, Iterable, Tuple, AsyncGenerator

from app.services.charts.builder import build_chart
from app.services.super_agent_wrapped import run_super_agent

CHART_CHUNK_SIZE = int(os.getenv("CHART_CHUNK_SIZE", "4096"))
_CHART_KEYWORDS = {"chart","candle","candlestick","ohlc","line","area","bar"}

def is_chart_intent(query: str) -> bool:
    q = (query or "").lower()
    return any(k in q for k in _CHART_KEYWORDS)

def _extract_symbol(q: str) -> Tuple[Optional[str], Optional[str]]:
    qn = (q or "").strip()
    m = re.search(r"\bfor\s+([A-Za-z0-9\.\-^ ]+)", qn, flags=re.IGNORECASE)
    raw = m.group(1).strip().strip("'\"") if m else qn
    raw = re.sub(r"\b(\d+[smhdw]|[0-9]+(d|mo|y)|\d+m)\b", "", raw, flags=re.IGNORECASE)
    for w in _CHART_KEYWORDS.union({"for","on","in","using","with"}):
        raw = re.sub(rf"\b{re.escape(w)}\b", "", raw, flags=re.IGNORECASE)
    raw = " ".join(raw.split())
    if not raw:
        return None, None
    idx_map = {"nifty": "^NSEI", "nifty 50": "^NSEI", "bank nifty": "^NSEBANK", "sensex": "^BSESN"}
    low = raw.lower()
    for k, v in idx_map.items():
        if k in low:
            return v, None
    if "." in raw or raw.startswith("^"):
        return raw.upper(), None
    tok = raw.split()[0].upper()
    return f"{tok}.NS", None

def _chunk_string(s: str, size: int) -> Iterable[str]:
    for i in range(0, len(s), size):
        yield s[i:i+size]

async def message_stream_chunks(message_pb2, request) -> AsyncGenerator[object, None]:
    """Async generator yielding MessageChunk responses for server streaming."""
    query = getattr(request, "query", "")
    chat_id = getattr(request, "chat_id", "") or None
    user_id = getattr(request, "user_id", "") or None

    if is_chart_intent(query):
        symbol, _ = _extract_symbol(query)
        payload = await build_chart(query=query, symbol=symbol, chart_type="", range_="", interval="", title="", description="")
        envelope = {"kind": "chart", "payload": payload}
        text = json.dumps(envelope, separators=(",", ":"))
        for chunk in _chunk_string(text, CHART_CHUNK_SIZE):
            yield message_pb2.MessageChunk(text=chunk, end=False, index=0)  # type: ignore
        return

    # Run the (sync) super agent off-thread to avoid blocking the event loop
    result = await asyncio.to_thread(run_super_agent, query, {"chat_id": chat_id, "user_id": user_id})
    out = (result.get("output") if isinstance(result, dict) else None) or str(result)
    idx = 0
    for chunk in _chunk_string(out, 2048):
        yield message_pb2.MessageChunk(text=chunk, end=False, index=idx)  # type: ignore
        idx += 1

async def handle_get_chart(message_pb2, request):
    query = getattr(request, "query", "") or ""
    symbol = getattr(request, "symbol", "") or None
    chart_type = getattr(request, "chart_type", "") or None
    range_hint = getattr(request, "range", "") or None
    interval_hint = getattr(request, "interval", "") or None
    payload = await build_chart(
        query=query,
        symbol= symbol or _extract_symbol(query)[0],
        chart_type= chart_type or "",
        range_= range_hint or "",
        interval= interval_hint or "",
        title="",
        description="",
    )
    return message_pb2.ChartResponse(  # type: ignore
        json=json.dumps(payload, separators=(",", ":")),
        content_type="application/json",
    )