from typing import Any, List
from langchain.tools import tool
from langchain_exa import ExaSearchResults, ExaFindSimilarResults, ExaSearchRetriever
from ..config import settings
from bs4 import BeautifulSoup
import requests


def _format_exa_result(result: Any) -> str:
    try:
        title = (result.get("title") or "").strip()
        url = (result.get("url") or "").strip()
        snippet = (result.get("text") or result.get("snippet") or "").strip()
        parts = [p for p in [title, url, snippet] if p]
        return "\n".join(parts)
    except Exception:
        return ""


@tool("exa_search")
def exa_search(query: str, max_results: int = 5) -> str:
    """
    Web search via EXA. Returns a newline-separated list of title, url, snippet blocks.
    """
    try:
        search = ExaSearchResults(api_key=settings.EXA_API_KEY, num_results=max_results)
        results: List[dict] = search.invoke({"query": query})  # type: ignore
        lines = []
        for r in results or []:
            fmt = _format_exa_result(r)
            if fmt:
                lines.append(fmt)
        return "\n\n".join(lines) or "No results."
    except Exception as e:
        return f"EXA search failed: {e}"


@tool("exa_find_similar")
def exa_find_similar(url_or_text: str, max_results: int = 5) -> str:
    """
    Find similar pages to the given URL or text using EXA.
    """
    try:
        finder = ExaFindSimilarResults(api_key=settings.EXA_API_KEY, num_results=max_results)
        results: List[dict] = finder.invoke({"query": url_or_text})  # type: ignore
        lines = []
        for r in results or []:
            fmt = _format_exa_result(r)
            if fmt:
                lines.append(fmt)
        return "\n\n".join(lines) or "No similar results."
    except Exception as e:
        return f"EXA find similar failed: {e}"


def _truncate(text: str, limit: int) -> str:
    if not text:
        return ""
    return text if len(text) <= limit else text[: limit - 3] + "..."


@tool("fetch_url")
def fetch_url(url: str, limit: int = 6000) -> str:
    """
    Fetch a URL and return visible text (truncated).
    """
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        text = " ".join(soup.get_text(separator=" ").split())
        return _truncate(text, limit)
    except Exception as e:
        return f"Fetch failed for {url}: {e}"


@tool("exa_live_search")
def exa_live_search(query: str, max_results: int = 5) -> str:
    """
    Retrieve live search results as short snippets using EXA retriever.
    """
    try:
        retriever = ExaSearchRetriever.from_api_key(
            api_key=settings.EXA_API_KEY,
            k=max_results,
            text=True,
        )
        docs = retriever.get_relevant_documents(query)
        lines = []
        for d in docs or []:
            title = (d.metadata.get("title") or "").strip()
            url = (d.metadata.get("url") or "").strip()
            snippet = _truncate(d.page_content or "", 400)
            block = "\n".join([p for p in [title, url, snippet] if p])
            if block:
                lines.append(block)
        return "\n\n".join(lines) or "No live results."
    except Exception as e:
        return f"EXA live search failed: {e}"


