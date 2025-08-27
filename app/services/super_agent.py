from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional, TypedDict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import asyncio
import re
import json
from urllib.parse import urlparse

import httpx
from langchain.schema import Document
from langgraph.graph import END, StateGraph

# Local modules
from .config import settings
from .azure_openai import chat_model
from .vector_store import news_store
from .rag import chunk_text, upsert_news, upsert_stocks
from .tools.exa import exa_search

try:
    from langchain_exa import ExaSearchRetriever
except Exception:  # pragma: no cover
    ExaSearchRetriever = None

try:
    from supabase import create_client
except Exception:  # pragma: no cover
    create_client = None

from langchain.prompts import ChatPromptTemplate
from bs4 import BeautifulSoup

# Constants
# Cooldown window for news freshness checks (avoid refetching/reenriching too often)
RECENCY_HOURS = 6
CURRENCY_SYMBOL = "₹"  # INR for Indian market

# ----------------------------
# State & Utilities
# ----------------------------

class AgentState(TypedDict, total=False):
    question: str
    company: Optional[str]
    ticker: Optional[str]
    exchange: Optional[Literal["NSE", "BSE"]]
    intent: Literal["stock", "news", "both"]
    news_detail: Literal["short", "medium", "long"]
    stock_data: Optional[Dict[str, Any]]
    news_items: Optional[List[Dict[str, Any]]]
    output: Optional[str]
    memory: Dict[str, Any]
    now: datetime


_INTENT_RE = {
    "stock": re.compile(r"\b(price|stock|quote|chart|market cap|marketcap|volume|pe|high|low|ohlc|today)\b", re.I),
    "news": re.compile(r"\b(news|headline|article|report|update|what\'s happening)\b", re.I),
}


def _parse_intent(q: str) -> Literal["stock", "news", "both"]:
    is_stock = bool(_INTENT_RE["stock"].search(q))
    is_news = bool(_INTENT_RE["news"].search(q))
    if is_stock and is_news:
        return "both"
    if is_stock:
        return "stock"
    if is_news:
        return "news"
    return "both"


_DETAIL_RE = {
    "long": re.compile(r"\b(detailed|long(\s+form)?|elaborate|deep\s*dive|comprehensive|full(\s+analysis)?)\b", re.I),
    "short": re.compile(r"\b(short|brief|tl;?dr|concise|summary)\b", re.I),
}


def _parse_news_detail(q: str) -> Literal["short", "medium", "long"]:
    if _DETAIL_RE["long"].search(q):
        return "long"
    if _DETAIL_RE["short"].search(q):
        return "short"
    return "medium"


# ----------------------------
# 1) Resolve Ticker
# ----------------------------

@dataclass
class TickerRecord:
    company_name: str
    nse_symbol: Optional[str]
    bse_symbol: Optional[str]


def _supabase_client():
    if create_client is None:
        return None
    try:
        return create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)
    except Exception:
        return None


def _fuzzy_match_company(name: str, rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    qnorm = _normalize_company_query(name)
    tokens = [t for t in re.split(r"\W+", qnorm) if t]
    if not tokens:
        return None

    best: Optional[Dict[str, Any]] = None
    best_score = -1.0
    qset = set(tokens)
    qphrase = " ".join(tokens)

    for r in rows:
        cname = (r.get("company_name") or "").strip().casefold()
        comp = _normalize_company_query(cname)
        if not comp:
            continue
        ctoks = [t for t in re.split(r"\W+", comp) if t]
        cset = set(ctoks)

        overlap = len(qset & cset)
        union = len(qset | cset) or 1
        jacc = overlap / union
        phrase = 2.0 if qphrase and qphrase in comp else 0.0
        prefix = 2.0 if comp.startswith(qphrase) else 0.0
        has_nse = 1.0 if r.get("nse_symbol") else 0.0
        has_bse = 0.5 if r.get("bse_symbol") else 0.0
        score = overlap * 3.0 + jacc * 5.0 + phrase + prefix + has_nse + has_bse

        if score > best_score:
            best = r
            best_score = score

    return best


def _normalize_company_query(name: str) -> str:
    n = (name or "").strip().casefold()
    n = re.sub(r"[\"'`]+", "", n)
    # Keep 'industry/industries' as they are meaningful (e.g., Reliance Industries)
    suffixes = r"\b(ltd\.?|limited|pvt\.?|private|inc\.?|co\.?|company|corp\.?|corporation)\b"
    n = re.sub(suffixes, "", n, flags=re.I)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def resolve_ticker_node(state: AgentState) -> AgentState:
    question = state.get("question", "")
    memory = state.get("memory", {})
    extract_prompt = ChatPromptTemplate.from_template(
        """Extract the primary Indian listed company name OR its NSE/BSE ticker from the query.
- If referring to prior context (e.g., 'it', 'this company'), output {{"entity": "NONE"}}.
- If it's an index like Nifty or Sensex, extract that.
- Output JSON: {{"entity": "exact name or ticker"}}. No extra text.
Examples:
Query: What's the price of Tata Motors?
Output: {{"entity": "Tata Motors"}}
Query: News on RELIANCE?
Output: {{"entity": "RELIANCE"}}
Query: How's it doing today?
Output: {{"entity": "NONE"}}

Query: {question}"""
    )
    try:
        resp = (extract_prompt | chat_model(temperature=0)).invoke({"question": question})
        data = json.loads(resp.content)
        name_or_ticker = data.get("entity")
        if name_or_ticker == "NONE":
            name_or_ticker = None
    except Exception:
        name_or_ticker = None

    if not name_or_ticker:
        if memory.get("ticker"):
            state["company"] = memory.get("company")
            state["ticker"] = memory.get("ticker")
            state["exchange"] = memory.get("exchange")
            return state
        y = _yahoo_search_symbol(question)
        if y and y.get("ticker"):
            state.update(y)
            memory.update(y)
            state["memory"] = memory
            return state
        state["company"] = None
        state["ticker"] = None
        state["exchange"] = None
        return state

    q_lower = f"{question} {name_or_ticker}".lower()
    index_map = [
        (re.compile(r"\b(nifty\s*50|nifty50|\bnifty\b)\b", re.I), {"ticker": "^NSEI", "company": "Nifty 50", "exchange": None}),
        (re.compile(r"\b(bank\s*nifty|nifty\s*bank)\b", re.I), {"ticker": "^NSEBANK", "company": "Nifty Bank", "exchange": None}),
        (re.compile(r"\bsensex\b", re.I), {"ticker": "^BSESN", "company": "Sensex", "exchange": None}),
        (re.compile(r"\b(nifty\s*next\s*50|niftynext50)\b", re.I), {"ticker": "^NSMIDCP", "company": "Nifty Next 50", "exchange": None}),
    ]
    for rx, val in index_map:
        if rx.search(q_lower):
            state.update(val)
            memory.update(val)
            state["memory"] = memory
            return state

    sb = _supabase_client()
    symbol = (name_or_ticker or "").upper().strip()

    row = None
    if sb:
        try:
            exact = sb.table("companies").select("company_name,nse_symbol,bse_symbol").or_(
                f"nse_symbol.eq.{symbol},bse_symbol.eq.{symbol}"
            ).limit(1).execute()
            row = exact.data[0] if exact.data else None
        except Exception:
            pass

    if not row and name_or_ticker and sb:
        try:
            qnorm = _normalize_company_query(name_or_ticker)
            toks = [t for t in re.split(r"\W+", qnorm) if t]
            # OR across top tokens for broader match
            clauses = [f"company_name.ilike.%{t}%" for t in toks[:3]]
            or_clause = ",".join(clauses) if clauses else f"company_name.ilike.%{qnorm}%"
            like = sb.table("companies").select("company_name,nse_symbol,bse_symbol").or_(or_clause).limit(100).execute()
            rows = like.data or []
            row = _fuzzy_match_company(name_or_ticker, rows)
        except Exception:
            pass

    if not row and name_or_ticker:
        query = f"{name_or_ticker} NSE or BSE ticker symbol site:Moneycontrol.com OR site:Economictimes.com India stock exchange"
        try:
            exa_res = exa_search.invoke({"query": query, "max_results": 3})
            extract_prompt = ChatPromptTemplate.from_template(
                """Extract ticker and company from results. Prefer NSE. JSON: {{"company": "name", "ticker": "SYMBOL", "exchange": "NSE" or "BSE"}} or nulls.
Results: {results}"""
            )
            resp = (extract_prompt | chat_model(temperature=0)).invoke({"results": exa_res})
            data = json.loads(resp.content)
            if data.get("company"):
                row = {
                    "company_name": data["company"],
                    "nse_symbol": data["ticker"] if data["exchange"] == "NSE" else None,
                    "bse_symbol": data["ticker"] if data["exchange"] == "BSE" else None,
                }
        except Exception:
            pass

    if not row:
        y = _yahoo_search_symbol(name_or_ticker)
        if y and y.get("ticker"):
            state.update(y)
            memory.update(y)
            state["memory"] = memory
            return state
        state["company"] = None
        state["ticker"] = None
        state["exchange"] = None
        return state

    rec = TickerRecord(
        company_name=row["company_name"],
        nse_symbol=row.get("nse_symbol"),
        bse_symbol=row.get("bse_symbol"),
    )

    if rec.nse_symbol:
        state["ticker"] = rec.nse_symbol
        state["exchange"] = "NSE"
    elif rec.bse_symbol:
        state["ticker"] = rec.bse_symbol
        state["exchange"] = "BSE"
    state["company"] = rec.company_name

    memory.update({
        "company": state["company"],
        "ticker": state["ticker"],
        "exchange": state["exchange"],
    })
    state["memory"] = memory
    return state


# ----------------------------
# 2) Stock Data
# ----------------------------

def _yf_symbol(ticker: str, exchange: Optional[str]) -> str:
    if exchange == "NSE":
        return f"{ticker}.NS"
    if exchange == "BSE":
        return f"{ticker}.BO"
    return ticker


def _yahoo_search_symbol(query: str) -> Optional[Dict[str, Optional[str]]]:
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
                    return {"company": company, "ticker": sym[:-3], "exchange": exc[:-1] + "E"}
        return None
    except Exception:
        return None


def _yahoo_chart(symbol: str, interval: str = "1m", range_: str = "1d") -> Dict[str, Any]:
    base = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {"interval": interval, "range": range_, "includePrePost": "true", "events": "div|split|earn"}
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.yahoo.com/"}
    try:
        r = httpx.get(base, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        return r.json()["chart"]["result"][0]
    except Exception:
        return {}


def _yahoo_quote(symbol: str) -> Dict[str, Any]:
    base = "https://query2.finance.yahoo.com/v7/finance/quote"
    params = {"symbols": symbol}
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.yahoo.com/"}
    try:
        r = httpx.get(base, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        return r.json()["quoteResponse"]["result"][0]
    except Exception:
        return {}


def _last_non_null(xs: List[Optional[float]]) -> Optional[float]:
    for v in reversed(xs or []):
        if v is not None:
            return float(v)
    return None


def _compute_snapshot_from_chart(chart: Dict[str, Any]) -> Dict[str, Any]:
    if not chart:
        return {}
    meta = chart.get("meta", {})
    q0 = chart.get("indicators", {}).get("quote", [{}])[0]
    last_price = _last_non_null(q0.get("close")) or meta.get("regularMarketPrice")
    day_high = max(q0.get("high") or [], default=None) or meta.get("regularMarketDayHigh")
    day_low = min(q0.get("low") or [], default=None) or meta.get("regularMarketDayLow")
    volume = sum(q0.get("volume") or []) or meta.get("regularMarketVolume")
    prev_close = meta.get("chartPreviousClose") or meta.get("previousClose") or meta.get("regularMarketPreviousClose")
    percent_change = ((last_price - prev_close) / prev_close * 100) if last_price and prev_close else None
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

    # Fallback: try to resolve from company if ticker missing
    if not ticker and company:
        y = _yahoo_search_symbol(company)
        if y and y.get("ticker"):
            state["ticker"] = ticker = y["ticker"]
            state["exchange"] = ex = y.get("exchange")
            state["company"] = y.get("company") or company

    # Fallback: try to resolve exchange if missing
    if ticker and not ex:
        y = _yahoo_search_symbol(ticker)
        if y and y.get("ticker"):
            state["ticker"] = ticker = y["ticker"]
            state["exchange"] = ex = y.get("exchange")

    if not ticker:
        state["stock_data"] = None
        return state

    symbol = _yf_symbol(ticker, ex)
    attempts = [("1m", "1d"), ("5m", "5d"), ("15m", "5d"), ("1d", "1mo")]
    snap = {}
    for interval, range_ in attempts:
        chart = _yahoo_chart(symbol, interval=interval, range_=range_)
        if chart:
            snap = _compute_snapshot_from_chart(chart)
            break

    if not snap:
        q = _yahoo_quote(symbol)
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


# ----------------------------
# 3) News
# ----------------------------

def _news_cache_lookup(query: str, filters: Dict[str, str]) -> List[Document]:
    """Retrieve potential cached news docs using vector similarity with metadata filter.

    filters should include either {"ticker": TICKER} or {"company": COMPANY} to scope results.
    """
    try:
        # Always scope to news documents
        f = dict(filters or {})
        f.setdefault("type", "news")
        return news_store().similarity_search(query or "news", k=8, filter=f)
    except Exception:
        return []


def _news_cache_is_fresh(docs: List[Document], now: datetime) -> bool:
    """Return True if any cached doc is within RECENCY_HOURS of now.

    This is the cooldown mechanism (6h by default) to avoid repeated fetch/upsert.
    """
    threshold = now - timedelta(hours=RECENCY_HOURS)
    for d in docs:
        ts = d.metadata.get("ts")
        if ts and datetime.fromisoformat(ts.replace("Z", "+00:00")) > threshold:
            return True
    return False


def _fetch_news_via_exa(query: str, k: int = 5) -> List[Dict[str, Any]]:
    if ExaSearchRetriever is None:
        return []
    retriever = ExaSearchRetriever(
        exa_api_key=settings.EXA_API_KEY,
        k=k,
        type="auto",
        livecrawl="always",
        text_contents_options={"max_characters": 2000},
        summary={"query": "Neutral factual sentence (20 words max) with key facts."},
    )
    try:
        docs = retriever.invoke(query)
    except Exception:
        return []
    items = []
    seen = set()
    for d in docs:
        md = d.metadata
        title = md.get("title", "Untitled").strip()
        url = md.get("url", "").strip()
        summary = md.get("summary", d.page_content[:200]).strip()
        key = (title.lower(), url.lower())
        if key in seen or not url:
            continue
        seen.add(key)
        items.append({"title": title, "url": url, "summary": summary})
    return items or [{"title": "No news", "url": "", "summary": ""}]


def get_news_node(state: AgentState) -> AgentState:
    now = state.get("now") or datetime.now(timezone.utc)
    company = state.get("company")
    ticker = state.get("ticker")
    query = company or ticker or ""
    filters = {"ticker": ticker} if ticker else {"company": company} if company else {}
    # Prefer grouped summarized docs first, then any summarized, then raw
    grouped_docs = _news_cache_lookup(query, {**filters, "llm_summarized": True, "grouped": True})
    summarized_docs = grouped_docs or _news_cache_lookup(query, {**filters, "llm_summarized": True})
    cached_docs = summarized_docs or _news_cache_lookup(query, filters)
    items: List[Dict[str, Any]] = []
    use_cache = bool(cached_docs) and _news_cache_is_fresh(cached_docs, now)

    if use_cache:
        # Sort summarized first, then by recency using epoch seconds
        cached_docs.sort(key=lambda d: (not d.metadata.get("llm_summarized", False), -_ts_to_epoch(d.metadata.get("ts"))))
        seen = set()
        for d in cached_docs:
            md = d.metadata
            key = ((md.get("url") or md.get("source") or "").lower(), (md.get("title") or "").lower())
            if key in seen:
                continue
            seen.add(key)
            summary = _strip_urls(d.page_content)
            items.append({"title": md.get("title", ""), "url": md.get("url", md.get("source", "")), "summary": summary})
            if len(items) >= 3:
                break
    else:
        # Fresh fetch only if cooldown expired or no cache
        exa_query = f"{query} latest India stock news"
        items = _fetch_news_via_exa(exa_query)

        if items:
            all_docs: List[Document] = []
            for it in items:
                content = f"{it['title']}\n{it['summary']}\n{it['url']}"
                meta = {"ticker": ticker or "", "company": company or "", "ts": now.isoformat(), "title": it['title'], "url": it['url'], "type": "news", "llm_summarized": False}
                all_docs.extend(chunk_text(content, meta))
            if all_docs:
                upsert_news(all_docs)

            # Synchronously produce a grouped, time-stamped summary with source links
            try:
                grouped = _summarize_news_items(items, company, ticker, now)
            except Exception:
                grouped = None
            if grouped and grouped.get("summary"):
                gmeta = {
                    "ticker": ticker or "",
                    "company": company or "",
                    "ts": now.isoformat(),
                    "title": grouped["title"],
                    "type": "news",
                    "llm_summarized": True,
                    "grouped": True,
                    "source_links": json.dumps(grouped.get("links", [])),
                }
                upsert_news(chunk_text(grouped["summary"], gmeta))
                # Prepend grouped summary to items so the user sees it immediately
                items = [{"title": grouped["title"], "url": grouped.get("links", [""])[0] if grouped.get("links") else "", "summary": grouped["summary"]}] + items[:2]
            # Optional background enrichment of top items
            try:
                asyncio.get_running_loop().create_task(_bg_enrich_news(items, company, ticker, now.isoformat()))
            except RuntimeError:
                pass

    state["news_items"] = items
    return state


def _has_recent_enriched(company: Optional[str], ticker: Optional[str], now: datetime) -> bool:
    try:
        filters = {"type": "news", "llm_summarized": True, **({"ticker": ticker} if ticker else {}), **({"company": company} if company else {})}
        docs = news_store().similarity_search(company or ticker or "news", k=5, filter=filters)
        threshold = now - timedelta(hours=RECENCY_HOURS * 2)
        for d in docs:
            ts = d.metadata.get("ts")
            if ts and datetime.fromisoformat(ts.replace("Z", "+00:00")) >= threshold:
                return True
    except Exception:
        return False
    return False


async def _bg_enrich_news(items: List[Dict[str, Any]], company: Optional[str], ticker: Optional[str], ts: str):
    headers = {"User-Agent": "Mozilla/5.0"}
    for it in items[:3]:
        try:
            url = it["url"]
            title = it["title"]
            content = it["summary"]
            if url:
                resp = httpx.get(url, headers=headers, timeout=8, follow_redirects=True)
                soup = BeautifulSoup(resp.text, "lxml")
                [tag.extract() for tag in soup(["script", "style"])]
                text = " ".join(soup.get_text().split())[:4000]
                content = f"{title}: {text}"
            prompt = ChatPromptTemplate.from_template(
                """Summarize to 2-4 factual sentences. Neutral. Include facts. End with [source].
Title: {title}\nContent: {content}\nSummary:"""
            )
            resp = (prompt | chat_model(temperature=0.1)).invoke({"title": title, "content": content})
            improved = resp.content.strip()
            if improved:
                meta = {"ticker": ticker or "", "company": company or "", "source": url, "title": title, "type": "news", "ts": ts, "llm_summarized": True}
                upsert_news(chunk_text(improved, meta))
        except Exception:
            pass


def _summarize_news_items(items: List[Dict[str, Any]], company: Optional[str], ticker: Optional[str], now: datetime) -> Optional[Dict[str, Any]]:
    """Create a concise market-impact summary from multiple items with source links."""
    if not items:
        return None
    # Build a compact plain-text input for the LLM
    lines = []
    links = []
    for it in items[:5]:
        t = (it.get("title") or "").strip()
        s = (it.get("summary") or "").strip()
        u = (it.get("url") or "").strip()
        if u:
            links.append(u)
        lines.append(f"- {t}: {s}")
    text = "\n".join(lines)[:4000]
    prompt = ChatPromptTemplate.from_template(
        """You are a market analyst. Summarize the following headlines into 3-5 bullet points:
Focus on: relevance, factual tone, sentiment (positive/negative/neutral), and potential near-term stock impact.
End with a one-line time-stamp like: [As of HH:MM IST, DD Mon YYYY]. Keep it under 120 words.
Company: {company}\nTicker: {ticker}\nItems:\n{text}\nSummary:"""
    )
    try:
        resp = (prompt | chat_model(temperature=0.1)).invoke({"company": company or "", "ticker": ticker or "", "text": text})
        summary = resp.content.strip()
        title = f"{(company or ticker or 'Market')} — Summary as of {now.astimezone(timezone(timedelta(hours=5, minutes=30))).strftime('%I:%M %p IST, %b %d, %Y')}"
        return {"summary": summary, "title": title, "links": links}
    except Exception:
        return None


# ----------------------------
# 4) Decider
# ----------------------------

def decider_node(state: AgentState) -> AgentState:
    q = state.get("question", "")
    state["intent"] = _parse_intent(q)
    state["news_detail"] = _parse_news_detail(q)
    return state


# ----------------------------
# 5) Merge & Format Output
# ----------------------------

def _fmt_money(x: Any) -> str:
    try:
        n = float(x)
        for unit in ["", "K", "M", "B", "T"]:
            if abs(n) < 1000:
                return f"{CURRENCY_SYMBOL}{n:,.2f}{unit}"
            n /= 1000
        return f"{CURRENCY_SYMBOL}{n:.2f}P"
    except Exception:
        return "-"


_URL_RE = re.compile(r"https?://\S+", re.I)


def _strip_urls(text: str) -> str:
    return re.sub(r"\s+", " ", _URL_RE.sub("", text)).strip()


def _ts_to_epoch(ts: Optional[str]) -> int:
    """Convert ISO timestamp to epoch seconds for sorting; return 0 if invalid."""
    if not ts:
        return 0
    try:
        return int(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp())
    except Exception:
        return 0


def _brand_from_url(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower().replace("www.", "").split(".")[0]
        return host or "link"
    except Exception:
        return "link"


def _limit_sentences(text: str, max_sentences: int, max_words: int) -> str:
    text = _strip_urls(text)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out = []
    wc = 0
    for s in sentences[:max_sentences]:
        words = s.split()
        if wc + len(words) > max_words:
            break
        out.append(" ".join(words))
        wc += len(words)
    summary = " ".join(out)
    if summary and not re.search(r"[.!?]$", summary):
        summary += "."
    return summary


def _change_emoji(pct: Optional[float]) -> str:
    if pct is None:
        return ""
    return "📈" if pct > 0 else "📉" if pct < 0 else "~"


# Replace the existing _format_stock_section function in super_agent.py
def _format_stock_section(company: str, ticker: str, ex: str, stock: Dict[str, Any]) -> str:
    if not stock:
        return ""
    price = _fmt_money(stock.get("current_price"))
    pct = stock.get("percent_change")
    pct_str = f"{pct:.2f}%" if pct is not None else "-"
    high = _fmt_money(stock.get("daily_high"))
    low = _fmt_money(stock.get("daily_low"))
    mcap = _fmt_money(stock.get("market_cap"))
    vol = _fmt_money(stock.get("volume"))
    emoji = _change_emoji(pct)

    # Strict Markdown table with precise end-of-line control
    if any(v is not None for v in [high, low, mcap, vol]):
        table = f"""**Stock Snapshot** for **{company} ({ticker} - {ex})**\n\n\
|---\n\
| **Metric**      | **Value**       |\n\
|-----------------|-----------------|\n\
| Price           | {price} {emoji} {pct_str} |\n\
| High / Low      | {high} / {low}  |\n\
| Market Cap      | {mcap}          |\n\
| Volume          | {vol}           |\n\
|---\n\n"""
    else:
        table = f"""**Stock Snapshot** for **{company} ({ticker} - {ex})**\n\n\
- **Price**: {price} {emoji} {pct_str}\n\n\
---\n"""

    # Debug: Print raw table to verify
    print(f"Generated table:\n{table}")
    return table


def _news_section(items: List[Dict[str, Any]], detail: Literal["short", "medium", "long"]) -> str:
    if not items:
        return "\n\n*No recent news available. Try again later!*"
    if detail == "short":
        max_sent, max_words = 1, 30
    elif detail == "long":
        max_sent, max_words = 5, 150
    else:
        max_sent, max_words = 3, 80
    bullets = []
    for it in items[:4]:
        title = it.get("title", "Untitled")
        summary = _limit_sentences(it.get("summary", ""), max_sent, max_words)
        url = it.get("url", "")
        if url:
            brand = _brand_from_url(url)
            tag = f" [{brand}]({url})"  # Clickable Markdown link
        else:
            tag = ""
        bullets.append(f"- **{title}**: {summary}{tag}")
    return f"\n\n**Recent News** ✨:\n" + "\n".join(bullets) + "\n---"


def merge_results_node(state: AgentState) -> AgentState:
    company = state.get("company") or "Market"
    ticker = state.get("ticker") or ""
    ex = state.get("exchange") or ""
    intent = state.get("intent", "both")
    stock = state.get("stock_data") or {}
    news = state.get("news_items") or []
    detail = state.get("news_detail", "medium")

    output = f"Here’s the scoop on {company}{f' ({ticker})' if ticker else ''} as of {datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime('%I:%M %p IST, %B %d, %Y')}:\n\n"

    if intent in ("stock", "both") and stock:
        output += _format_stock_section(company, ticker, ex, stock) + "\n"

    if intent in ("news", "both") and news:
        output += _news_section(news, detail)

    if not (stock or news):
        output = f"Oops! No data found for the Indian market. Try a specific NSE/BSE ticker like 'TATAMOTORS' or 'NIFTY'!\n"

    state["output"] = output
    return state


# ----------------------------
# Graph Assembly
# ----------------------------

def build_super_agent():
    sg = StateGraph(AgentState)
    sg.add_node("resolve_ticker", resolve_ticker_node)
    sg.add_node("decider", decider_node)
    sg.add_node("get_stock", get_stock_node)
    sg.add_node("get_news", get_news_node)
    sg.add_node("merge_results", merge_results_node)

    sg.set_entry_point("resolve_ticker")
    sg.add_edge("resolve_ticker", "decider")

    def route_from_decider(state: AgentState):
        intent = state["intent"]
        if intent == "stock":
            return "get_stock"
        if intent == "news":
            return "get_news"
        return "get_stock"

    sg.add_conditional_edges(
        "decider",
        route_from_decider,
        {"get_stock": "get_stock", "get_news": "get_news"},
    )

    def after_stock(state: AgentState):
        return "get_news" if state["intent"] == "both" else "merge_results"

    sg.add_conditional_edges("get_stock", after_stock, {"get_news": "get_news", "merge_results": "merge_results"})
    sg.add_edge("get_news", "merge_results")
    sg.add_edge("merge_results", END)

    return sg.compile()


# ----------------------------
# Public API
# ----------------------------

def run_super_agent(question: str, memory: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    graph = build_super_agent()
    now = datetime.now(timezone.utc)
    init: AgentState = {"question": question, "memory": memory or {}, "now": now}
    final_state: AgentState = graph.invoke(init)
    if memory is not None:
        # Persist resolved context for downstream calls
        if final_state.get("company"):
            memory["company"] = final_state.get("company")
        if final_state.get("ticker"):
            memory["ticker"] = final_state.get("ticker")
        if final_state.get("exchange"):
            memory["exchange"] = final_state.get("exchange")
    return final_state


# Lightweight helper: resolve only company/ticker/exchange using the same logic as the graph's first node.
def resolve_company_ticker(question: str, memory: Optional[Dict[str, Any]] = None) -> Dict[str, Optional[str]]:
    now = datetime.now(timezone.utc)
    state: AgentState = {"question": question, "memory": memory or {}, "now": now}
    state = resolve_ticker_node(state)
    mem = state.get("memory", memory or {})
    # return values and updated memory for caller to persist
    return {
        "company": state.get("company"),
        "ticker": state.get("ticker"),
        "exchange": state.get("exchange"),
        "memory": mem,
    }


def resolve_company_ticker_fast(query: str) -> Dict[str, Optional[str]]:
    """Resolve company/ticker/exchange quickly using Supabase fuzzy search; avoid LLM.

    Strategy:
    - Extract a likely company phrase from the query (after 'for', last quoted phrase, or whole query).
    - If explicit Yahoo-style symbol (contains '.' or startswith '^'), return it (no exchange mapping).
    - Query Supabase `companies` table with ilike OR across top tokens and fuzzy-score the best row; prefer NSE.
    - If Supabase unavailable or no match, fall back to Yahoo search API for the phrase.
    Returns keys: company, ticker, exchange (NSE/BSE) where possible.
    """
    s = (query or "").strip()
    # Pull phrase after 'for ...' or last quoted string
    m = re.search(r"\bfor\s+([A-Za-z0-9&\./\-\s]{2,60})", s, re.I)
    phrase = m.group(1).strip() if m else None
    if not phrase:
        qs = re.findall(r"[\"']([^\"']+)[\"']", s)
        phrase = (qs[-1].strip() if qs else s)
    # Clean trailing chart keywords
    phrase = re.sub(r"\s+(?:on|in|with|using|chart|line|area|bar|candle|candlestick|ohlc)\b.*$", "", phrase, flags=re.I).strip()
    # If looks like explicit symbol/index
    if "." in phrase or phrase.startswith("^"):
        sym = phrase.upper()
        if sym.endswith(".NS"):
            return {"company": None, "ticker": sym[:-3], "exchange": "NSE"}
        if sym.endswith(".BO"):
            return {"company": None, "ticker": sym[:-3], "exchange": "BSE"}
        return {"company": None, "ticker": sym, "exchange": None}

    sb = _supabase_client()
    row = None
    if sb:
        try:
            qnorm = _normalize_company_query(phrase)
            toks = [t for t in re.split(r"\W+", qnorm) if t]
            clauses = [f"company_name.ilike.%{t}%" for t in toks[:3]] or [f"company_name.ilike.%{qnorm}%"]
            like = sb.table("companies").select("company_name,nse_symbol,bse_symbol").or_(",".join(clauses)).limit(100).execute()
            rows = like.data or []
            row = _fuzzy_match_company(phrase, rows)
        except Exception:
            row = None

    if row:
        ticker = row.get("nse_symbol") or row.get("bse_symbol")
        exch: Optional[str] = "NSE" if row.get("nse_symbol") else ("BSE" if row.get("bse_symbol") else None)
        return {"company": row.get("company_name"), "ticker": ticker, "exchange": exch}

    # Fallback: Yahoo symbol search
    y = _yahoo_search_symbol(phrase)
    if y and y.get("ticker"):
        return {"company": y.get("company"), "ticker": y.get("ticker"), "exchange": y.get("exchange")}

    return {"company": None, "ticker": None, "exchange": None}