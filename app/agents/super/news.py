from __future__ import annotations
from typing import Any, Dict, List, Optional, Literal
from datetime import datetime, timedelta, timezone
import json
import asyncio

import httpx
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate

from app.tools.config import settings
from app.tools.azure_openai import chat_model
try:
    from app.tools.vector_store import news_store  # type: ignore
except Exception:  # pragma: no cover
    news_store = None  # type: ignore
try:
    from app.tools.rag import chunk_text, upsert_news  # type: ignore
except Exception:  # pragma: no cover
    chunk_text = upsert_news = None  # type: ignore

try:
    from langchain_exa import ExaSearchRetriever
except Exception:  # pragma: no cover
    ExaSearchRetriever = None

from .state import AgentState
from .utils import ts_to_epoch, strip_urls, brand_from_url, RECENCY_HOURS


def _news_cache_lookup(query: str, filters: Dict[str, str]) -> List[Document]:
    try:
        if news_store is None:
            return []
        f = dict(filters or {})
        f.setdefault("type", "news")
        return news_store().similarity_search(query or "news", k=8, filter=f)
    except Exception:
        return []


def _news_cache_is_fresh(docs: List[Document], now: datetime) -> bool:
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


def _summarize_news_items(items: List[Dict[str, Any]], company: Optional[str], ticker: Optional[str], now: datetime) -> Optional[Dict[str, Any]]:
    if not items:
        return None
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


def get_news_node(state: AgentState) -> AgentState:
    now = state.get("now") or datetime.now(timezone.utc)
    company = state.get("company")
    ticker = state.get("ticker")
    query = company or ticker or ""
    filters = {"ticker": ticker} if ticker else {"company": company} if company else {}
    grouped_docs = _news_cache_lookup(query, {**filters, "llm_summarized": True, "grouped": True})
    summarized_docs = grouped_docs or _news_cache_lookup(query, {**filters, "llm_summarized": True})
    cached_docs = summarized_docs or _news_cache_lookup(query, filters)
    items: List[Dict[str, Any]] = []
    use_cache = bool(cached_docs) and _news_cache_is_fresh(cached_docs, now)

    if use_cache:
        cached_docs.sort(key=lambda d: (not d.metadata.get("llm_summarized", False), -ts_to_epoch(d.metadata.get("ts"))))
        seen = set()
        for d in cached_docs:
            md = d.metadata
            key = ((md.get("url") or md.get("source") or "").lower(), (md.get("title") or "").lower())
            if key in seen:
                continue
            seen.add(key)
            summary = strip_urls(d.page_content)
            items.append({"title": md.get("title", ""), "url": md.get("url", md.get("source", "")), "summary": summary})
            if len(items) >= 3:
                break
    else:
        exa_query = f"{query} latest India stock news"
        items = _fetch_news_via_exa(exa_query)

        if items:
            all_docs: List[Document] = []
            for it in items:
                content = f"{it['title']}\n{it['summary']}\n{it['url']}"
                meta = {"ticker": ticker or "", "company": company or "", "ts": now.isoformat(), "title": it['title'], "url": it['url'], "type": "news", "llm_summarized": False}
                all_docs.extend(chunk_text(content, meta))
            if all_docs and upsert_news:
                upsert_news(all_docs)

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
                if chunk_text and upsert_news:
                    upsert_news(chunk_text(grouped["summary"], gmeta))
                items = [{"title": grouped["title"], "url": grouped.get("links", [""])[0] if grouped.get("links") else "", "summary": grouped["summary"]}] + items[:2]
            try:
                asyncio.get_running_loop().create_task(_bg_enrich_news(items, company, ticker, now.isoformat()))
            except RuntimeError:
                pass

    state["news_items"] = items
    return state
