from typing import List, Dict
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool

# Prefer the agent_tools implementations the app already uses
from .exa import (
    exa_search as _exa_search,
    exa_find_similar as _exa_find_similar,
    fetch_url as _fetch_url,
    exa_live_search as _exa_live_search,
)


# ---------- Optional Structured inputs (Pydantic schemas) ----------
class ExaSearchInput(BaseModel):
    query: str = Field(..., description="Search query")
    max_results: int = Field(5, ge=1, le=20, description="Maximum number of results")


class ExaFindSimilarInput(BaseModel):
    url_or_text: str = Field(..., description="URL or text to find similar pages for")
    max_results: int = Field(5, ge=1, le=20, description="Maximum number of results")


class FetchUrlInput(BaseModel):
    url: str = Field(..., description="Web page URL to fetch")
    max_chars: int = Field(800, ge=100, le=4000, description="Maximum characters in the snippet")


class ExaLiveSearchInput(BaseModel):
    query: str = Field(..., description="Live search query")
    k: int = Field(8, ge=1, le=20, description="Number of documents to retrieve")
    max_chars: int = Field(600, ge=100, le=4000, description="Maximum characters in each summary")


# ---------- Non-structured tool objects (as defined with @tool) ----------
ALL_TOOLS = {
    "exa_search": _exa_search,
    "exa_find_similar": _exa_find_similar,
    "fetch_url": _fetch_url,
    "exa_live_search": _exa_live_search,
}


# ---------- StructuredTool wrappers ----------
STRUCTURED_TOOLS = {
    "exa_search": StructuredTool.from_function(
        func=lambda query, max_results=5: _exa_search.func(query, max_results),
        name="exa_search",
        description="Search the web with EXA for the given query. Returns brief results list.",
        args_schema=ExaSearchInput,
    ),
    "exa_find_similar": StructuredTool.from_function(
        func=lambda url_or_text, max_results=5: _exa_find_similar.func(url_or_text, max_results),
        name="exa_find_similar",
        description="Find web pages similar to the given URL or text using EXA.",
        args_schema=ExaFindSimilarInput,
    ),
    "fetch_url": StructuredTool.from_function(
        func=lambda url, max_chars=800: _fetch_url.func(url, max_chars),
        name="fetch_url",
        description="Fetch a web page and return a concise text snippet (title + summary).",
        args_schema=FetchUrlInput,
    ),
    "exa_live_search": StructuredTool.from_function(
        func=lambda query, k=8, max_chars=600: _exa_live_search.func(query, k, max_chars),
        name="exa_live_search",
        description="Live-crawl search with EXA that returns concise, recent summaries.",
        args_schema=ExaLiveSearchInput,
    ),
}


# ---------- Categories / Use-cases ----------
TOOL_CATEGORIES: Dict[str, List[str]] = {
    "web_search": ["exa_search", "exa_live_search", "fetch_url"],
    "similarity": ["exa_find_similar"],
}


def load_tools(use_cases: List[str] | None = None, structured: bool = False):
    """Return a list of tools based on use-cases. If structured=True, return StructuredTools.

    use_cases examples: ["web_search"], ["similarity"], ["web_search", "similarity"].
    If None or empty, returns all tools.
    """
    name_to_tool = STRUCTURED_TOOLS if structured else ALL_TOOLS

    if not use_cases:
        return [name_to_tool[n] for n in ALL_TOOLS.keys() if n in name_to_tool]

    selected: List[str] = []
    for uc in use_cases:
        names = TOOL_CATEGORIES.get(uc, [])
        selected.extend(names)
    # Preserve order and uniqueness based on ALL_TOOLS order
    seen = set()
    ordered = []
    for n in ALL_TOOLS.keys():
        if n in selected and n not in seen and n in name_to_tool:
            ordered.append(name_to_tool[n])
            seen.add(n)
    return ordered


