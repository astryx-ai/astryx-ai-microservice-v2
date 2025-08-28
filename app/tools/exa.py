from __future__ import annotations
from typing import Optional
from langchain_core.tools import tool
import requests
import os


EXA_API_KEY = os.getenv("EXA_API_KEY")


@tool
def exa_search(query: str, max_results: int = 5) -> str:
    """
    Web search via EXA. Returns a newline-separated list of title, url, snippet blocks.
    """
    try:
        url = "https://api.exa.ai/search"
        headers = {"x-api-key": EXA_API_KEY or "", "content-type": "application/json"}
        payload = {"query": query, "maxResults": int(max_results or 5)}
        r = requests.post(url, json=payload, headers=headers, timeout=12)
        r.raise_for_status()
        data = r.json()
        lines = []
        for item in (data.get("results") or [])[: int(max_results or 5)]:
            title = (item.get("title") or "").strip()
            link = (item.get("url") or "").strip()
            snippet = (item.get("text") or item.get("snippet") or "").strip()
            lines.append(title)
            if link:
                lines.append(link)
            if snippet:
                lines.append(snippet)
            lines.append("")
        return "\n".join(lines).strip()
    except Exception as e:
        return f"EXA search failed: {e}"


@tool
def exa_find_similar(url_or_text: str, max_results: int = 5) -> str:
    """
    Find similar pages to the given URL or text using EXA.
    """
    try:
        url = "https://api.exa.ai/findSimilar"
        headers = {"x-api-key": EXA_API_KEY or "", "content-type": "application/json"}
        payload = {"text": url_or_text, "maxResults": int(max_results or 5)}
        r = requests.post(url, json=payload, headers=headers, timeout=12)
        r.raise_for_status()
        data = r.json()
        lines = []
        for item in (data.get("results") or [])[: int(max_results or 5)]:
            title = (item.get("title") or "").strip()
            link = (item.get("url") or "").strip()
            snippet = (item.get("text") or item.get("snippet") or "").strip()
            lines.append(title)
            if link:
                lines.append(link)
            if snippet:
                lines.append(snippet)
            lines.append("")
        return "\n".join(lines).strip()
    except Exception as e:
        return f"EXA findSimilar failed: {e}"


@tool
def exa_live_search(query: str, max_results: int = 5) -> str:
    """Live crawl using EXA and return blocks of title, url, snippet."""
    try:
        url = "https://api.exa.ai/livecrawl"
        headers = {"x-api-key": EXA_API_KEY or "", "content-type": "application/json"}
        payload = {"query": query, "maxResults": int(max_results or 5), "liveCrawl": True}
        r = requests.post(url, json=payload, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
        lines = []
        for item in (data.get("results") or [])[: int(max_results or 5)]:
            title = (item.get("title") or "").strip()
            link = (item.get("url") or "").strip()
            snippet = (item.get("text") or item.get("snippet") or "").strip()
            lines.append(title)
            if link:
                lines.append(link)
            if snippet:
                lines.append(snippet)
            lines.append("")
        return "\n".join(lines).strip()
    except Exception as e:
        return f"EXA live search failed: {e}"


@tool
def fetch_url(url: str, max_chars: int = 2000) -> str:
    """Fetch the given URL and return the first max_chars characters of clean text."""
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        t = r.text
        return t[: int(max_chars or 2000)]
    except Exception as e:
        return f"fetch_url failed: {e}"
