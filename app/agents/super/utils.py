from __future__ import annotations
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse
import re

import httpx

CURRENCY_SYMBOL = "₹"
RECENCY_HOURS = 6

_URL_RE = re.compile(r"https?://\S+", re.I)

INTENT_RE = {
    "stock": re.compile(r"\b(price|stock|quote|chart|market cap|marketcap|volume|pe|high|low|ohlc|today)\b", re.I),
    "news": re.compile(r"\b(news|headline|article|report|update|what\'s happening)\b", re.I),
}

DETAIL_RE = {
    "long": re.compile(r"\b(detailed|long(\s+form)?|elaborate|deep\s*dive|comprehensive|full(\s+analysis)?)\b", re.I),
    "short": re.compile(r"\b(short|brief|tl;?dr|concise|summary)\b", re.I),
}


def parse_intent(q: str) -> str:
    is_stock = bool(INTENT_RE["stock"].search(q))
    is_news = bool(INTENT_RE["news"].search(q))
    if is_stock and is_news:
        return "both"
    if is_stock:
        return "stock"
    if is_news:
        return "news"
    return "both"


# Lightweight greeting detector
_GREETING_RE = re.compile(r"\b(hi|hello|hey|how\s*are\s*you|good\s*(morning|afternoon|evening)|what's\s*up|yo)\b", re.I)


def _fallback_llm_intent(query: str, memory: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Few-shot LLM fallback: returns a label or None if unavailable/failed.
    Allowed labels: greeting | stock | news | both | clarify
    """
    try:
        from langchain.prompts import ChatPromptTemplate  # type: ignore
        from app.tools.azure_openai import chat_model  # type: ignore
    except Exception:
        return None
    labels = "greeting | stock | news | both | clarify"
    mem_snip = ""
    try:
        # Keep memory compact in prompt
        if memory:
            parts = []
            for k in ("company","ticker","exchange"):
                v = memory.get(k)
                if v:
                    parts.append(f"{k}={v}")
            mem_snip = ", ".join(parts)
    except Exception:
        mem_snip = ""
    prompt = ChatPromptTemplate.from_template(
        """Classify user intent into exactly one of: {labels}.
Conversation so far (key facts): {memory}
Message: {query}
Answer with just one label from {labels} and nothing else."""
    )
    try:
        resp = (prompt | chat_model(temperature=0)).invoke({"labels": labels, "memory": mem_snip, "query": query})
        raw = (resp.content or "").strip().lower()
        raw = re.sub(r"[^a-z]", "", raw)
        if raw in {"greeting","stock","news","both","clarify"}:
            return raw
    except Exception:
        return None
    return None


def classify_intent(q: str, memory: Optional[Dict[str, Any]] = None) -> str:
    """Classify intent using message and memory.

    Returns: greeting | stock | news | both | clarify
    Rules:
    - greeting if small talk
    - pronoun reference with no context -> clarify
    - rule-based parse_intent for stock/news/both
    - if ambiguous or low-confidence, try LLM fallback
    - if still ambiguous, return clarify
    """
    s = (q or "").strip()
    if not s:
        return "clarify"
    if _GREETING_RE.search(s):
        return "greeting"
    base = parse_intent(s)

    # Pronoun reference without context
    pron = bool(re.search(r"\b(it|this|that|same|them)\b", s, re.I))
    has_ctx = bool(memory and (memory.get("ticker") or memory.get("company")))
    if pron and not has_ctx:
        # Try LLM once; else clarify
        llm = _fallback_llm_intent(s, memory)
        return llm or "clarify"

    if base == "both":
        # If no context, try LLM; otherwise accept both
        if not has_ctx:
            llm = _fallback_llm_intent(s, memory)
            if llm in {"stock","news","both"}:
                return llm
            return "clarify"
        return "both"

    # base is stock or news
    return base


def parse_news_detail(q: str) -> str:
    if DETAIL_RE["long"].search(q):
        return "long"
    if DETAIL_RE["short"].search(q):
        return "short"
    return "medium"


def strip_urls(text: str) -> str:
    return re.sub(r"\s+", " ", _URL_RE.sub("", text)).strip()


def ts_to_epoch(ts: Optional[str]) -> int:
    if not ts:
        return 0
    try:
        return int(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp())
    except Exception:
        return 0


def brand_from_url(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower().replace("www.", "").split(".")[0]
        return host or "link"
    except Exception:
        return "link"


def fmt_money(x: Any) -> str:
    try:
        n = float(x)
        for unit in ["", "K", "M", "B", "T"]:
            if abs(n) < 1000:
                return f"{CURRENCY_SYMBOL}{n:,.2f}{unit}"
            n /= 1000
        return f"{CURRENCY_SYMBOL}{n:.2f}P"
    except Exception:
        return "-"


def change_emoji(pct: Optional[float]) -> str:
    if pct is None:
        return ""
    return "📈" if pct > 0 else "📉" if pct < 0 else "~"


def yf_symbol(ticker: str, exchange: Optional[str]) -> str:
    if exchange == "NSE":
        return f"{ticker}.NS"
    if exchange == "BSE":
        return f"{ticker}.BO"
    return ticker


def yahoo_search_symbol(query: str) -> Optional[Dict[str, Optional[str]]]:
    if not query.strip():
        return None
    base = "https://query2.finance.yahoo.com/v1/finance/search"
    params = {"q": query, "quotesCount": 5, "newsCount": 0}
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.yahoo.com/"}
    try:
        r = httpx.get(base, params=params, headers=headers, timeout=8)
        r.raise_for_status()
        quotes = r.json().get("quotes", [])
        for exc in ("NS", "BO"):
            for q in quotes:
                if q.get("quoteType") != "EQUITY":
                    continue
                sym = q.get("symbol", "").upper()
                if sym.endswith(f".{exc}"):
                    company = q.get("shortname") or q.get("longname") or ""
                    return {"company": company, "ticker": sym[:-3], "exchange": ("NSE" if exc=="NS" else "BSE")}
        return None
    except Exception:
        return None


def yahoo_autocomplete(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Yahoo query1 autocomplete with rich fields; returns Indian (NSE/BSE) equities only.

    Each candidate: {
      company, ticker, exchange, symbol, score, shortname, longname, sector, industry
    }
    """
    q = (query or "").strip()
    if not q:
        return []
    base = "https://query1.finance.yahoo.com/v1/finance/search"
    params = {
        "q": q,
        "lang": "en-US",
        "region": "US",
        "quotesCount": max(1, min(limit, 20)),
        "newsCount": 0,
        "listsCount": 0,
        "enableFuzzyQuery": "false",
        "quotesQueryId": "tss_match_phrase_query",
        "multiQuoteQueryId": "multi_quote_single_token_query",
        "enableCb": "false",
        "enableNavLinks": "true",
        "enableEnhancedTrivialQuery": "true",
        "enableLogoUrl": "true",
        "enableLists": "false",
        "recommendCount": 5,
        "enablePrivateCompany": "true",
    }
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.yahoo.com/"}
    try:
        r = httpx.get(base, params=params, headers=headers, timeout=8)
        r.raise_for_status()
        data = r.json() or {}
        out: List[Dict[str, Any]] = []
        for q in data.get("quotes", []) or []:
            if q.get("quoteType") != "EQUITY":
                continue
            sym = (q.get("symbol") or "").upper()
            exc = None
            if sym.endswith(".NS"):
                exc = "NSE"
            elif sym.endswith(".BO"):
                exc = "BSE"
            if not exc:
                continue
            out.append({
                "company": q.get("longname") or q.get("shortname") or "",
                "ticker": sym[:-3],
                "exchange": exc,
                "symbol": sym,
                "score": q.get("score"),
                "shortname": q.get("shortname"),
                "longname": q.get("longname"),
                "sector": q.get("sector") or q.get("sectorDisp"),
                "industry": q.get("industry") or q.get("industryDisp"),
                "exchDisp": q.get("exchDisp"),
            })
        # Deduplicate by symbol while preserving order
        seen = set()
        uniq: List[Dict[str, Any]] = []
        for it in out:
            if it["symbol"] in seen:
                continue
            seen.add(it["symbol"])
            uniq.append(it)
        return uniq
    except Exception:
        return []


def _norm_company(s: str) -> str:
    import re as _re
    n = (s or "").strip().casefold()
    n = _re.sub(r"[\"'`]+", "", n)
    n = _re.sub(r"\b(ltd\.?|limited|pvt\.?|private|inc\.?|co\.?|company|corp\.?|corporation)\b", "", n)
    n = _re.sub(r"\s+", " ", n).strip()
    return n


def yahoo_resolve_symbol(query: str) -> Optional[Dict[str, Any]]:
    """Direct, dependency-free ticker resolver using Yahoo autocomplete.

    Returns on success:
    {
      'symbol': 'RELIANCE.NS',
      'exchange': 'NSE',
      'longname': 'Reliance Industries Limited',
      'sector': 'Energy',
      'industry': 'Oil & Gas Integrated'
    }
    """
    q = (query or "").strip()
    if not q:
        return None
    # Explicit symbol given
    up = q.upper()
    if up.endswith('.NS') or up.endswith('.BO'):
        sym = up
        exch = 'NSE' if sym.endswith('.NS') else 'BSE'
        base = yahoo_autocomplete(sym[:-3], limit=5)
        # try to enrich
        meta = next((c for c in base if (c.get('symbol') or '').upper() == sym), None)
        return {
            'symbol': sym,
            'exchange': exch,
            'longname': (meta or {}).get('longname') or (meta or {}).get('company') or '',
            'sector': (meta or {}).get('sector') or '',
            'industry': (meta or {}).get('industry') or '',
        }
    # Otherwise autocomplete and rank
    ph = q
    norm = _norm_company(ph)
    cands = yahoo_autocomplete(ph, limit=8)
    if not cands:
        return None
    best = None
    best_score = -1.0
    single = len(norm.split()) == 1
    for c in cands:
        cname = (c.get('longname') or c.get('company') or '').strip()
        compn = _norm_company(cname)
        # simple similarity
        lev = 0.0
        try:
            # local levenshtein-light
            from difflib import SequenceMatcher
            lev = SequenceMatcher(None, norm, compn).ratio()
        except Exception:
            lev = 0.0
        contains = 0.8 if norm and norm in compn else 0.0
        prefix = 0.8 if compn.startswith(norm) else 0.0
        exb = 0.5 if c.get('exchange') == 'NSE' else (0.3 if c.get('exchange') == 'BSE' else 0.0)
        yscore = float(c.get('score') or 0.0) / 10000.0
        parent = 0.0
        if single:
            if 'industries' in compn:
                parent += 0.6
            if any(x in compn for x in ['power','green','ports','capital','transmission','energy']):
                parent -= 0.2
        s = lev * 5.0 + contains + prefix + exb + yscore + parent
        if s > best_score:
            best, best_score = c, s
    if not best:
        return None
    sym = best.get('symbol') or ''
    exch = 'NSE' if sym.upper().endswith('.NS') else ('BSE' if sym.upper().endswith('.BO') else best.get('exchange'))
    return {
        'symbol': sym,
        'exchange': exch,
        'longname': best.get('longname') or best.get('company') or '',
        'sector': best.get('sector') or '',
        'industry': best.get('industry') or '',
    }


def yahoo_chart(symbol: str, interval: str = "1m", range_: str = "1d") -> Dict[str, Any]:
    base = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {"interval": interval, "range": range_, "includePrePost": "true", "events": "div|split|earn"}
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.yahoo.com/"}
    try:
        r = httpx.get(base, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        return r.json()["chart"]["result"][0]
    except Exception:
        return {}


def yahoo_quote(symbol: str) -> Dict[str, Any]:
    base = "https://query2.finance.yahoo.com/v7/finance/quote"
    params = {"symbols": symbol}
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.yahoo.com/"}
    try:
        r = httpx.get(base, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        return r.json()["quoteResponse"]["result"][0]
    except Exception:
        return {}


def last_non_null(xs: List[Optional[float]]) -> Optional[float]:
    for v in reversed(xs or []):
        if v is not None:
            return float(v)
    return None
