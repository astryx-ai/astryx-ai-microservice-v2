# app/services/agent_tools/registry.py
from typing import List, Dict
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool

from .exa import exa_search_func, exa_find_similar_func, fetch_url_func, exa_live_search_func
from .chart import CHART_GENERATE_TOOL, chart_run

# ----------------- Pydantic schemas -----------------
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

# ----------------- Tool dictionaries -----------------
ALL_TOOLS = {
    "exa_search": exa_search_func,
    "exa_find_similar": exa_find_similar_func,
    "fetch_url": fetch_url_func,
    "exa_live_search": exa_live_search_func,
    # Fix: chart_run returns dict already suitable for /agent/stream
    "chart_generate": chart_run,
}


STRUCTURED_TOOLS = {
    "exa_search": StructuredTool.from_function(func=exa_search_func, name="exa_search", description="Search the web with EXA for the given query.", args_schema=ExaSearchInput),
    "exa_find_similar": StructuredTool.from_function(func=exa_find_similar_func, name="exa_find_similar", description="Find web pages similar to the given URL or text using EXA.", args_schema=ExaFindSimilarInput),
    "fetch_url": StructuredTool.from_function(func=fetch_url_func, name="fetch_url", description="Fetch a web page and return a concise text snippet.", args_schema=FetchUrlInput),
    "exa_live_search": StructuredTool.from_function(func=exa_live_search_func, name="exa_live_search", description="Live search using EXA.", args_schema=ExaLiveSearchInput),
    "chart_generate": CHART_GENERATE_TOOL,
}

# ----------------- Tool categories -----------------
TOOL_CATEGORIES: Dict[str, List[str]] = {
    "web_search": ["exa_search", "exa_live_search", "fetch_url"],
    "similarity": ["exa_find_similar"],
    "charts": ["chart_generate"],
}

def load_tools(use_cases: List[str] | None = None, structured: bool = False):
    """Return a list of tools based on use-cases."""
    name_to_tool = STRUCTURED_TOOLS if structured else ALL_TOOLS
    if not use_cases:
        return [name_to_tool[n] for n in ALL_TOOLS.keys() if n in name_to_tool]

    selected: List[str] = []
    for uc in use_cases:
        names = TOOL_CATEGORIES.get(uc, [])
        selected.extend(names)

    seen = set()
    ordered = []
    for n in ALL_TOOLS.keys():
        if n in selected and n not in seen and n in name_to_tool:
            ordered.append(name_to_tool[n])
            seen.add(n)
    return ordered
