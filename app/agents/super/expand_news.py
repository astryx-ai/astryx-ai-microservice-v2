from __future__ import annotations
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import json
import httpx
from bs4 import BeautifulSoup
from langchain.prompts import ChatPromptTemplate

from app.tools.azure_openai import chat_model
from app.tools.rag import chunk_text, upsert_news
from app.tools.vector_store import news_store


def _fetch_url_text(url: str, max_chars: int = 4000) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = httpx.get(url, headers=headers, timeout=12, follow_redirects=True)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        [t.extract() for t in soup(["script", "style", "noscript"])]
        text = " ".join(soup.get_text().split())
        return text[: max_chars]
    except Exception:
        return ""


def expand_news_tool(*, url: str, company: Optional[str], ticker: Optional[str]) -> Dict[str, Any]:
    """Fetch article via URL (from Exa or user), summarize with OpenAI, and store (summary+link) in VectorDB.

    Returns: { summary: str, link: str, stored: bool }
    """
    link = (url or "").strip()
    if not link:
        return {"summary": "", "link": "", "stored": False}
    raw = _fetch_url_text(link)
    if not raw:
        raw = link  # at least keep URL as context
    prompt = ChatPromptTemplate.from_template(
        """Summarize the following article in 3-6 concise bullet points.
Avoid fluff, keep it factual and useful for an investor tracking {company} ({ticker}).
Add a final line: 'Source: {link}'. Keep under 120 words.

Article:
{text}
"""
    )
    try:
        resp = (prompt | chat_model(temperature=0.1)).invoke({
            "company": company or "",
            "ticker": ticker or "",
            "text": raw,
            "link": link,
        })
        summary = (resp.content or "").strip()
    except Exception:
        summary = raw[:240] + ("..." if len(raw) > 240 else "")
    # store in VectorDB
    ts = datetime.now(timezone.utc).isoformat()
    meta = {
        "ticker": (ticker or "").upper(),
        "company": company or "",
        "url": link,
        "source": link,
        "title": "Expanded article summary",
        "type": "news",
        "llm_summarized": True,
        "ts": ts,
        "expanded": True,
    }
    try:
        docs = chunk_text(summary, meta)
        if docs:
            upsert_news(docs)
        stored = bool(docs)
    except Exception:
        stored = False
    return {"summary": summary, "link": link, "stored": stored}
