from typing import Any, List
from langchain.tools import tool
from langchain_exa import ExaSearchResults, ExaFindSimilarResults, ExaSearchRetriever
from app.config import settings
import httpx
from bs4 import BeautifulSoup
from app.utils.stream_utils import emit_process
from .helper_tools import needs_recency_injection, inject_datetime_into_query


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
        return "\n".join(lines)
    except Exception:
        return str(result)[:600]


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _extract_key_sentences(text: str, max_sentences: int = 5) -> str:
    """Return up to max_sentences that contain numbers or key indicators."""
    import re
    sentences = re.split(r"(?<=[.!?])\s+", text or "")
    picked: List[str] = []
    for s in sentences:
        if re.search(r"\d", s) or re.search(r"\b(USD|Rs|INR|%|million|billion|crore|lakh)\b", s, re.I):
            picked.append(s.strip())
        if len(picked) >= max_sentences:
            break
    return " ".join(picked)


@tool("exa_search")
def exa_search(query: str, max_results: int = 5) -> str:
    """Search the web and return detailed results that preserve key facts and numbers."""
    print(f"[Tool] exa_search called | query='{query}', max_results={max_results}")
    emit_process({"message": f"Searching Internet for '{query}'"})
    try:
        # Optionally inject current date/time into the query if user intent implies recency
        effective_query = inject_datetime_into_query(query) if needs_recency_injection(query) else query
        if effective_query != query:
            print(f"[Tool] exa_search recency injection -> '{effective_query}'")
        retriever = ExaSearchRetriever(
            exa_api_key=settings.EXA_API_KEY,
            k=max(1, min(int(max_results), 10)),
            type="auto",
            text_contents_options={"max_characters": 50000},
            # Disable summarization to avoid overly concise outputs
            summary=None,
        )
        docs = retriever.invoke(effective_query)
        try:
            print(f"[Tool] exa_search retrieved {len(docs)} documents")
        except Exception:
            print("[Tool] exa_search retrieved documents (unknown length)")
        lines: List[str] = []
        for i, d in enumerate(docs, 1):
            meta = getattr(d, "metadata", {}) or {}
            title = meta.get("title") or ""
            url = meta.get("url") or meta.get("id") or ""
            # Prefer full content if available, fall back to metadata summary
            summary = (d.page_content or "").strip() or meta.get("summary") or ""
            head = (title or url).strip()
            lines.append(f"{i}. {head} ({url})\n   {summary}")
        return "\n".join(lines) if lines else _format_exa_result([])
    except Exception:
        # Fallback to simple result list
        t = ExaSearchResults(exa_api_key=settings.EXA_API_KEY, max_results=max_results)
        result = t.invoke(query)
        return _format_exa_result(result)


@tool("exa_find_similar")
def exa_find_similar(url_or_text: str, max_results: int = 5) -> str:
    """Find similar pages and include descriptive snippets with key details."""
    print(f"[Tool] exa_find_similar called | max_results={max_results}")
    emit_process({"message": "Finding similar pages"})
    try:
        t = ExaFindSimilarResults(exa_api_key=settings.EXA_API_KEY, max_results=max_results)
        result = t.invoke(url_or_text)
        print(f"[Tool] exa_find_similar got result type={type(result)}")
        items: List[Any] = []
        if isinstance(result, dict) and isinstance(result.get("results"), list):
            items = result["results"]
        elif isinstance(result, list):
            items = result
        lines: List[str] = []
        for i, it in enumerate(items[: max_results], 1):
            url = (it.get("url") if isinstance(it, dict) else getattr(it, "url", None)) or (
                it.get("id") if isinstance(it, dict) else getattr(it, "id", "")
            )
            title = (it.get("title") if isinstance(it, dict) else getattr(it, "title", "")) or url
            snippet = ""
            if url and i <= 3:  # enrich top-3 with a fetch snippet
                try:
                    snippet = fetch_url.func(url, max_chars=600)
                except Exception:
                    snippet = ""
            if snippet:
                lines.append(f"{i}. {title} ({url})\n   {snippet}")
            else:
                lines.append(f"{i}. {title} ({url})")
        return "\n".join(lines) if lines else _format_exa_result(result)
    except Exception:
        return _format_exa_result([])


@tool("fetch_url")
def fetch_url(url: str, max_chars: int = 800) -> str:
    """Fetch a web page and return a concise text snippet (title + summary)."""
    print(f"[Tool] fetch_url called | url='{url}', max_chars={max_chars}")
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
        # Try to include numeric sentences for richer detail
        numeric_bits = _extract_key_sentences(desc, max_sentences=4)
        combined = f"{desc} {numeric_bits}".strip()
        snippet = _truncate(combined, max_chars)
        head = title or url
        return f"{head}: {snippet}"
    except Exception as e:
        return f"Failed to fetch {url}: {e}"


@tool("fetch_url_text")
def fetch_url_text(url: str, chunk_index: int = 1, chunk_size: int = 4000, max_total_chars: int = 120000) -> str:
    """Fetch a web page and return raw text in chunks without summarization.

    - chunk_index: 1-based index of the chunk to return
    - chunk_size: approximate number of characters per chunk
    - max_total_chars: safety cap on total extracted characters
    Returns: "Chunk i/N | <title> (<url>)\n<text>"
    """
    print(f"[Tool] fetch_url_text called | url='{url}', chunk_index={chunk_index}, chunk_size={chunk_size}, max_total_chars={max_total_chars}")
    # Emit only the link being fetched, suppress chunk numbers in user-visible stream
    try:
        emit_process({"message": f"Fetching: {url}"})
    except Exception:
        pass
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; AstryxBot/1.0)"}
        resp = httpx.get(url, headers=headers, timeout=20.0, follow_redirects=True)
        resp.raise_for_status()
        html = resp.text
        soup = BeautifulSoup(html, "lxml")
        title = (soup.title.string if soup.title else "").strip()
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "svg", "img"]):
            try:
                tag.extract()
            except Exception:
                pass
        text = " ".join((soup.get_text(separator=" ") or "").split())
        if not text:
            return f"Chunk {chunk_index}/1 | {title or url} ({url})\n"
        # Enforce safety cap
        if len(text) > max_total_chars:
            text = text[:max_total_chars]
        if chunk_size <= 0:
            chunk_size = 4000
        # Compute chunks (1-based)
        total_chunks = max(1, (len(text) + chunk_size - 1) // chunk_size)
        idx = max(1, min(int(chunk_index), total_chunks))
        start = (idx - 1) * chunk_size
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]
        head = (title or url).strip()
        return f"Chunk {idx}/{total_chunks} | {head} ({url})\n{chunk_text}"
    except Exception as e:
        return f"Failed to fetch chunk {chunk_index} for {url}: {e}"

@tool("exa_live_search")
def exa_live_search(query: str, k: int = 8, max_chars: int = 1000) -> str:
    """Live-crawl search with EXA that returns detailed, recent summaries (title, URL, rich details)."""
    print(f"[Tool] exa_live_search called | query='{query}', k={k}, max_chars={max_chars}")
    emit_process({"message": f"Live searching Internet for '{query}'"})
    try:
        # Optionally inject current date/time into the query if user intent implies recency
        effective_query = inject_datetime_into_query(query) if needs_recency_injection(query) else query
        if effective_query != query:
            print(f"[Tool] exa_live_search recency injection -> '{effective_query}'")
        retriever = ExaSearchRetriever(
            exa_api_key=settings.EXA_API_KEY,
            k=max(1, min(int(k), 20)),
            type="auto",
            livecrawl="always",
            text_contents_options={"max_characters": 50000},
            # Disable summarization to preserve details
            summary=None,
        )
        docs = retriever.invoke(effective_query)
        try:
            print(f"[Tool] exa_live_search retrieved {len(docs)} documents")
        except Exception:
            print("[Tool] exa_live_search retrieved documents (unknown length)")
        lines: List[str] = []
        for i, d in enumerate(docs, 1):
            meta = getattr(d, "metadata", {}) or {}
            title = meta.get("title") or ""
            url = meta.get("url") or meta.get("id") or ""
            content = (d.page_content or "").strip()
            summary = content or meta.get("summary") or ""
            head = (title or url).strip()
            if head:
                lines.append(f"{i}. {head} ({url})\n   {summary}")
            else:
                lines.append(f"{i}. {summary}")
        return "\n".join(lines)
    except Exception as e:
        return f"EXA live search failed: {e}"


