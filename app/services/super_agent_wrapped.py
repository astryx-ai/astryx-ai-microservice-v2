from __future__ import annotations
from typing import Any, Dict, List
import re, requests
from langchain.schema import Document

from .super_agent import run_super_agent as _core_run_super_agent
from .rag import chunk_text, upsert_news, upsert_stocks
from .tools.exa import exa_live_search, exa_search  # langchain tools (@tool)

def _call_tool(tool, *args, **kwargs) -> str:
    try:
        return tool.func(*args, **kwargs)  # type: ignore[attr-defined]
    except Exception:
        return tool(*args, **kwargs)

def _parse_exa_blocks(text: str) -> List[Dict[str, str]]:
    blocks = []
    for block in (text or "").split("\n\n"):
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if not lines:
            continue
        title = lines[0]
        url = lines[1] if len(lines) > 1 and lines[1].startswith("http") else ""
        snippet = "\n".join(lines[2:]) if url else "\n".join(lines[1:])
        blocks.append({"title": title, "url": url, "snippet": snippet})
    return blocks

def _news_intent(q: str) -> bool:
    ql = (q or "").lower()
    return any(k in ql for k in ["news","headline","article","latest"])

def _stock_intent(q: str) -> bool:
    ql = (q or "").lower()
    return any(k in ql for k in ["price","quote","stock","chart","candlestick","ohlc"])

def _extract_symbol(q: str) -> str | None:
    m = re.search(r"\bfor\s+([A-Za-z0-9\.\-^ ]+)", q or "", flags=re.IGNORECASE)
    if m:
        raw = m.group(1).strip().strip("'\"")
    else:
        raw = (q or "").strip()
    raw = re.sub(r"\b(\d+[smhdw]|[0-9]+(d|mo|y)|\d+m)\b", "", raw, flags=re.IGNORECASE)
    for w in ["for","on","in","using","with","price","quote","stock","chart","candle","candlestick","ohlc","line","area","bar"]:
        raw = re.sub(rf"\b{re.escape(w)}\b", "", raw, flags=re.IGNORECASE)
    raw = " ".join(raw.split())
    if not raw:
        return None
    if "." in raw or raw.startswith("^"):
        return raw.upper()
    return f"{raw.split()[0].upper()}.NS"

def _fetch_yahoo_bars(symbol: str, range_: str = "5d", interval: str = "15m") -> list[dict]:
    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        "interval": interval,
        "range": range_,
        "includePrePost": "true",
        "events": "div|split|earn",
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    j = r.json()
    res = j.get("chart", {}).get("result", []) or []
    if not res:
        return []
    r0 = res[0]
    ts = r0.get("timestamp") or []
    ind = r0.get("indicators", {}).get("quote", [{}])[0] or {}
    opens = (ind.get("open") or [])
    highs = (ind.get("high") or [])
    lows = (ind.get("low") or [])
    closes = (ind.get("close") or [])
    vols = (ind.get("volume") or [])
    rows = []
    for i, t in enumerate(ts):
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        if o is None and h is None and l is None and c is None:
            continue
        rows.append({
            "ts": int(t),
            "open": o, "high": h, "low": l, "close": c,
            "volume": vols[i] if i < len(vols) else None,
        })
    return rows

def _upsert_news_for_query(query: str) -> None:
    try:
        text = _call_tool(exa_live_search, query, 5)
        if not text or "failed" in text.lower():
            text = _call_tool(exa_search, query, 5)
        blocks = _parse_exa_blocks(text)
        docs: List[Document] = []
        for b in blocks:
            md = {"url": b.get("url",""), "title": b.get("title",""), "source": b.get("url","")}
            docs.extend(chunk_text(b.get("snippet",""), md))
        upsert_news(docs)
    except Exception as e:
        print(f"[SuperAgent] news upsert skipped: {e}")

def _upsert_stocks_for_query(query: str) -> None:
    try:
        sym = _extract_symbol(query)
        if not sym:
            return
        rows = _fetch_yahoo_bars(sym, "5d", "15m") or _fetch_yahoo_bars(sym, "1mo", "1d")
        docs: List[Document] = []
        for r in rows:
            ts = int(r["ts"])
            md = {"ticker": sym, "ts": ts}
            text = f"{sym} {ts} O{r['open']} H{r['high']} L{r['low']} C{r['close']} V{r.get('volume')}"
            docs.append(Document(page_content=text, metadata=md))
        upsert_stocks(docs)
    except Exception as e:
        print(f"[SuperAgent] stock upsert skipped: {e}")

def run_super_agent(query: str, memory: Dict[str, Any] | None = None) -> Dict[str, Any]:
    result = _core_run_super_agent(query, memory=memory or {})
    if _news_intent(query):
        _upsert_news_for_query(query)
    if _stock_intent(query):
        _upsert_stocks_for_query(query)
    if isinstance(result, dict):
        return result
    return {"output": str(result)}