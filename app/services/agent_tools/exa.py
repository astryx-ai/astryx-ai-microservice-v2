from typing import Any, List
from langchain.tools import tool
from langchain_exa import ExaSearchResults, ExaFindSimilarResults, ExaSearchRetriever
from app.config import settings
import httpx
from bs4 import BeautifulSoup
from app.stream_utils import emit_process


def _format_exa_result(result: Any) -> str:
    try:
        # Normalize to a list of dicts if possible
        items: List[Any]
        if isinstance(result, dict) and "results" in result and isinstance(result["results"], list):
            items = result["results"]
        elif isinstance(result, list):
            items = result
        else:
            items = [result]

        lines = []
        for i, item in enumerate(items, 1):
            title = None
            url = None
            if isinstance(item, dict):
                title = item.get("title")
                url = item.get("url") or item.get("id")
            else:
                title = getattr(item, "title", None)
                url = getattr(item, "url", None) or getattr(item, "id", None)
            if title or url:
                lines.append(f"{i}. {title or ''} {('(' + url + ')') if url else ''}".strip())
        return "\n".join(lines[:10])
    except Exception:
        return str(result)[:600]


def exa_search_func(query: str, max_results: int = 5) -> str:
    """Search the web with EXA for the given query. Returns brief results list."""
    emit_process({"message": f"Searching Internet for '{query}'"})
    t = ExaSearchResults(exa_api_key=settings.EXA_API_KEY, max_results=max_results)
    result = t.invoke(query)
    return _format_exa_result(result)


def exa_find_similar_func(url_or_text: str, max_results: int = 5) -> str:
    """Find web pages similar to the given URL or text using EXA. Returns brief results list."""
    emit_process({"message": "Finding similar pages"})
    t = ExaFindSimilarResults(exa_api_key=settings.EXA_API_KEY, max_results=max_results)
    result = t.invoke(url_or_text)
    return _format_exa_result(result)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def fetch_url_func(url: str, max_chars: int = 800) -> str:
    """Fetch a web page and return a concise text snippet (title + summary)."""
    emit_process({"message": f"Fetching {url}"})
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; AstryxBot/1.0)"}
        resp = httpx.get(url, headers=headers, timeout=12.0, follow_redirects=True)
        resp.raise_for_status()
        html = resp.text
        soup = BeautifulSoup(html, "lxml")
        # Prefer meta description
        title = (soup.title.string if soup.title else "").strip()
        meta = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
        desc = meta["content"].strip() if meta and meta.get("content") else ""
        # Fallback extract
        if not desc:
            for tag in soup(["script", "style", "noscript"]):
                tag.extract()
            text = " ".join(soup.get_text(separator=" ").split())
            desc = text
        snippet = _truncate(desc, max_chars)
        head = title or url
        return f"{head}: {snippet}"
    except Exception as e:
        return f"Failed to fetch {url}: {e}"


def exa_live_search_func(query: str, k: int = 8, max_chars: int = 600) -> str:
    """Live-crawl search with EXA that returns concise, recent summaries (title, URL, brief)."""
    emit_process({"message": f"Live searching Internet for '{query}'"})
    try:
        retriever = ExaSearchRetriever(
            exa_api_key=settings.EXA_API_KEY,
            k=max(1, min(int(k), 20)),
            type="auto",
            livecrawl="always",
            text_contents_options={"max_characters": 3000},
            summary={"query": "generate one line summary in simple words."},
        )
        docs = retriever.invoke(query)
        lines: List[str] = []
        for i, d in enumerate(docs[: min(5, len(docs))], 1):
            meta = getattr(d, "metadata", {}) or {}
            title = meta.get("title") or ""
            url = meta.get("url") or meta.get("id") or ""
            summary = meta.get("summary") or " ".join(d.page_content.split())[:max_chars]
            summary = _truncate(summary, max_chars)
            head = (title or url).strip()
            if head:
                lines.append(f"{i}. {head} ({url})\n   {summary}")
            else:
                lines.append(f"{i}. {summary}")
        return "\n".join(lines)
    except Exception as e:
        return f"EXA live search failed: {e}"


# Optional: also expose tool-decorated variants (not used by structured registry)
exa_search = tool("exa_search")(exa_search_func)
exa_find_similar = tool("exa_find_similar")(exa_find_similar_func)
fetch_url = tool("fetch_url")(fetch_url_func)
exa_live_search = tool("exa_live_search")(exa_live_search_func)

