import json, re, os, asyncio
from typing import Dict, Any, Optional, Iterable, Tuple, AsyncGenerator, List
from datetime import datetime, timezone, timedelta

from app.services.charts.builder import build_chart
from app.services.super_agent_wrapped import run_super_agent
from app.services.super_agent import resolve_company_ticker
from app.services.memory_store import global_memory_store

CHART_CHUNK_SIZE = int(os.getenv("CHART_CHUNK_SIZE", "4096"))
TEXT_SENTENCE_CHUNK_SIZE = int(os.getenv("TEXT_SENTENCE_CHUNK_SIZE", "600"))
_CHART_KEYWORDS = {"chart","candle","candlestick","ohlc","line","area","bar"}

# Simple in-memory per-chat memory store
_MEMORY: Dict[str, Dict[str, Any]] = {}
_LAST_CONTEXT: Dict[str, Any] = {"company": None, "ticker": None, "exchange": None, "ts": None}
MEMORY_TTL_SEC = int(os.getenv("MEMORY_TTL_SEC", "900"))  # 15 minutes
_STORE = global_memory_store()

def _mem_key(chat_id: Optional[str], user_id: Optional[str]) -> Optional[str]:
    return chat_id or user_id

def is_chart_intent(query: str) -> bool:
    q = (query or "").lower()
    return any(k in q for k in _CHART_KEYWORDS)

def _extract_symbol(q: str) -> Tuple[Optional[str], bool]:
    qn = (q or "").strip()
    m = re.search(r"\bfor\s+([A-Za-z0-9\.\-^ ]+)", qn, flags=re.IGNORECASE)
    raw = m.group(1).strip().strip("'\"") if m else qn
    raw = re.sub(r"\b(\d+[smhdw]|[0-9]+(d|mo|y)|\d+m)\b", "", raw, flags=re.IGNORECASE)
    for w in _CHART_KEYWORDS.union({"for","on","in","using","with"}):
        raw = re.sub(rf"\b{re.escape(w)}\b", "", raw, flags=re.IGNORECASE)
    raw = " ".join(raw.split())
    # Strip obvious stray punctuation from ends
    raw = raw.strip("?,.!:;()[]{}\"'`")
    if not raw:
        return None, False
    idx_map = {"nifty": "^NSEI", "nifty 50": "^NSEI", "bank nifty": "^NSEBANK", "sensex": "^BSESN"}
    low = raw.lower()
    for k, v in idx_map.items():
        if k in low:
            return v, False
    # If user typed an explicit Yahoo symbol/index, accept after validation
    if "." in raw or raw.startswith("^"):
        if not re.search(r"[A-Za-z0-9]", raw):
            return None, False
        return raw.upper(), False
    # If phrase contains multiple words (likely a company name), don't guess a ticker here; let resolver handle it
    if " " in raw:
        return None, True
    tok = raw.split()[0].upper()
    # Validate token (must contain at least one alphanumeric)
    if not re.search(r"[A-Z0-9]", tok):
        return None, False
    # We are guessing based on a single token without explicit exchange; mark as guessed
    return f"{tok}.NS", True

def _chunk_string(s: str, size: int) -> Iterable[str]:
    for i in range(0, len(s), size):
        yield s[i:i+size]

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

def _chunk_sentences(text: str, max_len: int) -> Iterable[str]:
    """Yield chunks that end at sentence boundaries, up to max_len each."""
    text = (text or "").strip()
    if not text:
        return []
    # Split on sentence boundaries (simple heuristic)
    sentences: List[str] = re.split(r"(?<=[.!?])\s+", text)
    buf: List[str] = []
    cur_len = 0
    for s in sentences:
        if not s:
            continue
        if cur_len + len(s) + (1 if buf else 0) > max_len and buf:
            yield " ".join(buf)
            buf = [s]
            cur_len = len(s)
        else:
            buf.append(s)
            cur_len += len(s) + (1 if buf else 0)
    if buf:
        yield " ".join(buf)

async def message_stream_chunks(message_pb2, request) -> AsyncGenerator[object, None]:
    """Async generator yielding MessageChunk responses for server streaming."""
    query = getattr(request, "query", "")
    chat_id = getattr(request, "chat_id", "") or None
    user_id = getattr(request, "user_id", "") or None

    if is_chart_intent(query):
        symbol, guessed = _extract_symbol(query)
        resolved_via = "explicit" if (symbol and not guessed) else None
        # Use memory symbol if missing
        key = _mem_key(chat_id, user_id)
        if not symbol or guessed:
            # Prefer persisted memory, then in-proc cache
            mem = _STORE.load(chat_id, user_id, ttl_seconds=MEMORY_TTL_SEC) or (_MEMORY.get(key, {}) if (key and key in _MEMORY) else {})
            # fallback to last context if within TTL
            try:
                ts = _LAST_CONTEXT.get("ts")
                fresh = bool(ts and (datetime.now(timezone.utc) - datetime.fromisoformat(ts)).total_seconds() <= MEMORY_TTL_SEC)
            except Exception:
                fresh = False
            if not mem and fresh:
                mem = {k: _LAST_CONTEXT.get(k) for k in ("company","ticker","exchange")}
            # Try robust resolver against DB/Yahoo given the free-form query
            try:
                ctx = resolve_company_ticker(query, mem)
                tkr = ctx.get("ticker")
                ex = ctx.get("exchange")
                comp = ctx.get("company")
                # Update memory and persist if we found anything
                if comp or tkr or ex:
                    mem.update({k: v for k, v in {"company": comp, "ticker": tkr, "exchange": ex}.items() if v})
                    if key is not None:
                        _MEMORY[key] = mem
                    _STORE.save(chat_id, user_id, mem)
                if tkr and ex:
                    symbol = f"{tkr}.NS" if ex == "NSE" else (f"{tkr}.BO" if ex == "BSE" else tkr)
                    resolved_via = "resolver"
                elif tkr:
                    symbol = tkr
                    resolved_via = "resolver"
            except Exception:
                # Fall back to any prior memory if resolver fails
                tkr = (mem or {}).get("ticker")
                ex = (mem or {}).get("exchange")
                if tkr and ex:
                    symbol = f"{tkr}.NS" if ex == "NSE" else (f"{tkr}.BO" if ex == "BSE" else tkr)
                    resolved_via = "memory"
                elif tkr:
                    symbol = tkr
                    resolved_via = "memory"
        if not resolved_via and symbol and guessed:
            resolved_via = "guess"
        payload = await build_chart(query=query, symbol=symbol, chart_type="", range_="", interval="", title="", description="")
        # Attach minimal context for debugging client-side
        try:
            ctx_meta = {
                "resolved_via": resolved_via or "unknown",
                "chat_id": chat_id,
                "user_id": user_id,
                "symbol": symbol,
            }
        except Exception:
            ctx_meta = {"resolved_via": resolved_via or "unknown", "symbol": symbol}
        envelope = {"kind": "chart", "payload": payload, "context": ctx_meta}
        text = json.dumps(envelope, separators=(",", ":"))
        for chunk in _chunk_string(text, CHART_CHUNK_SIZE):
            yield message_pb2.MessageChunk(text=chunk, end=False, index=0)  # type: ignore
        return

    # Run the (sync) super agent off-thread to avoid blocking the event loop
    key = _mem_key(chat_id, user_id)
    # seed memory from persistent store; fallback to in-proc and last context
    mem = _STORE.load(chat_id, user_id, ttl_seconds=MEMORY_TTL_SEC) or _MEMORY.get(key or "", {})
    if not mem:
        try:
            ts = _LAST_CONTEXT.get("ts")
            if ts and (datetime.now(timezone.utc) - datetime.fromisoformat(ts)).total_seconds() <= MEMORY_TTL_SEC:
                mem.update({k: _LAST_CONTEXT.get(k) for k in ("company","ticker","exchange")})
        except Exception:
            pass
    result = await asyncio.to_thread(run_super_agent, query, mem)
    # Persist updated memory for this chat/user
    if key is not None:
        _MEMORY[key] = mem
    _STORE.save(chat_id, user_id, mem)
    # update last context snapshot
    try:
        comp = result.get("company") if isinstance(result, dict) else None
        tkr = result.get("ticker") if isinstance(result, dict) else None
        ex = result.get("exchange") if isinstance(result, dict) else None
        if comp or tkr or ex:
            _LAST_CONTEXT.update({
                "company": comp or _LAST_CONTEXT.get("company"),
                "ticker": tkr or _LAST_CONTEXT.get("ticker"),
                "exchange": ex or _LAST_CONTEXT.get("exchange"),
                "ts": datetime.now(timezone.utc).isoformat(),
            })
    except Exception:
        pass
    out = (result.get("output") if isinstance(result, dict) else None) or str(result)
    idx = 0
    for chunk in _chunk_sentences(out, TEXT_SENTENCE_CHUNK_SIZE):
        yield message_pb2.MessageChunk(text=chunk, end=False, index=idx)  # type: ignore
        idx += 1

async def handle_get_chart(message_pb2, request):
    query = getattr(request, "query", "") or ""
    symbol = getattr(request, "symbol", "") or None
    chart_type = getattr(request, "chart_type", "") or None
    range_hint = getattr(request, "range", "") or None
    interval_hint = getattr(request, "interval", "") or None
    chat_id = getattr(request, "chat_id", "") or None
    user_id = getattr(request, "user_id", "") or None
    # Fallback to resolver/memory if no explicit symbol provided
    if not symbol:
        resolved_via = None
        # Prefer robust resolver using the query first
        mem = _STORE.load(chat_id, user_id, ttl_seconds=MEMORY_TTL_SEC) or {}
        try:
            ctx = resolve_company_ticker(query, mem)
            tkr = ctx.get("ticker")
            ex = ctx.get("exchange")
            comp = ctx.get("company")
            if comp or tkr or ex:
                mem.update({k: v for k, v in {"company": comp, "ticker": tkr, "exchange": ex}.items() if v})
                _STORE.save(chat_id, user_id, mem)
            if tkr and ex:
                symbol = f"{tkr}.NS" if ex == "NSE" else (f"{tkr}.BO" if ex == "BSE" else tkr)
                resolved_via = "resolver"
            elif tkr:
                symbol = tkr
                resolved_via = "resolver"
        except Exception:
            pass
        # Then use last-context snapshot if still unresolved
        if not symbol:
            try:
                ts = _LAST_CONTEXT.get("ts")
                if ts and (datetime.now(timezone.utc) - datetime.fromisoformat(ts)).total_seconds() <= MEMORY_TTL_SEC:
                    tkr = _LAST_CONTEXT.get("ticker")
                    ex = _LAST_CONTEXT.get("exchange")
                    if tkr and ex:
                        symbol = f"{tkr}.NS" if ex == "NSE" else (f"{tkr}.BO" if ex == "BSE" else tkr)
                        resolved_via = "last_context"
                    elif tkr:
                        symbol = tkr
                        resolved_via = "last_context"
            except Exception:
                pass
    else:
        resolved_via = "explicit"
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