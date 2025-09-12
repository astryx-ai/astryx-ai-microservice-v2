from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool

# Import tool implementations (tools-only here)
from app.agent_tools.exa import (
    exa_search as _exa_search,
    exa_live_search as _exa_live_search,
    fetch_url_text as _fetch_url_text,
)
from app.agent_tools.chart_emit import emit_chart as _emit_chart
from app.agent_tools.chart_emit import ChartPayloadInput as _ChartPayloadInput
from app.agent_tools.financial_extraction import (
    shareholding_pattern_tool,
    ShareholdingPatternInput,
    fundamentals_header_tool,
    FundamentalsHeaderInput,
)
from app.agent_tools.price_movement import (
    price_movement_tool,
    PriceMovementInput,
)

# Note: subgraphs are not tools


# ---------- Pydantic schemas for structured tools ----------
# (Financial tool schemas imported from financial_extraction module)

class ExaSearchInput(BaseModel):
    query: str = Field(..., description="Search query")
    max_results: int = Field(5, ge=1, le=20, description="Maximum number of results")


class ExaLiveSearchInput(BaseModel):
    query: str = Field(..., description="Live search query")
    k: int = Field(8, ge=1, le=20, description="Number of documents to retrieve")
    max_chars: int = Field(
        1000, ge=100, le=8000, description="Maximum characters in each summary"
    )


class FetchUrlTextInput(BaseModel):
    url: str = Field(..., description="Web page URL to fetch in raw text chunks")
    chunk_index: int = Field(
        1, ge=1, le=1000, description="1-based index of chunk to return"
    )
    chunk_size: int = Field(
        4000, ge=500, le=20000, description="Approximate characters per chunk"
    )
    max_total_chars: int = Field(
        120000,
        ge=5000,
        le=500000,
        description="Safety cap on total extracted characters",
    )


class ChartEmitInput(BaseModel):
    # Backward compatibility shim if referenced elsewhere
    payload: Dict[str, Any] | None = None


# ---------- Core tools (non-structured) ----------

# Description with non-working graph
# description=(
#     "Emit a chart_data event directly to the stream. Pass one argument named 'payload'"
#     " which is a JSON object containing the chart specification. You MAY call this tool"
#     " multiple times in a single run to emit multiple charts. Supported types include:"
#     " bar-standard, bar-multiple, bar-stacked, bar-negative, area-standard, area-linear,"
#     " area-stacked, line-standard, line-linear, line-multiple, line-dots, line-label,"
#     " pie-standard, pie-label, pie-interactive, pie-donut, radar-standard, radar-lines-only,"
#     " radar-multiple, radial-standard, radial-stacked, radial-progress. The payload must include"
#     " the required keys for the chosen type (e.g., title, description, dataKey/nameKey or groupedKeys,"
#     " and data: []). Returns empty string."
# ),

ALL_TOOLS = {
    "exa_search": _exa_search,
    "exa_live_search": _exa_live_search,
    "fetch_url_text": _fetch_url_text,
    "emit_chart": StructuredTool.from_function(
        func=lambda payload: _emit_chart(payload),
        name="emit_chart",
        description=(
            "Emit a chart_data event directly to the stream. Pass one argument named 'payload'"
            " which is a JSON object containing the chart specification. You MAY call this tool"
            " multiple times in a single run to emit multiple charts. Supported types include:"
            " bar-standard, bar-stacked, bar-negative, area-standard, area-linear,"
            " area-stacked, line-standard, line-linear, line-label,"
            " pie-standard, pie-label, pie-interactive, pie-donut, radar-standard, radar-lines-only,"
            " radial-standard, radial-stacked, radial-progress. The payload must include"
            " the required keys for the chosen type (e.g., title, description, dataKey/nameKey or groupedKeys,"
            " and data: []). Returns empty string."
        ),
        args_schema=_ChartPayloadInput,
    ),
}


# ---------- Structured tools ----------
STRUCTURED_TOOLS = {
    "exa_search": StructuredTool.from_function(
        func=lambda query, max_results=5: _exa_search.func(query, max_results),
        name="exa_search",
        description="Search the web and return detailed results preserving all key facts and numbers.",
        args_schema=ExaSearchInput,
    ),
    "exa_live_search": StructuredTool.from_function(
        func=lambda query, k=8, max_chars=1000: _exa_live_search.func(
            query, k, max_chars
        ),
        name="exa_live_search",
        description="Live-crawl search that returns detailed summaries preserving key facts and numbers.",
        args_schema=ExaLiveSearchInput,
    ),
    "fetch_url_text": StructuredTool.from_function(
        func=lambda url, chunk_index=1, chunk_size=4000, max_total_chars=120000: _fetch_url_text.func(
            url, chunk_index, chunk_size, max_total_chars
        ),
        name="fetch_url_text",
        description="Fetch a web page and return raw text in chunks without summarization.",
        args_schema=FetchUrlTextInput,
    ),
    "emit_chart": StructuredTool.from_function(
        func=lambda payload: _emit_chart(payload),
        name="emit_chart",
        description="Emit a chart_data event from a generic payload after schema validation. Returns empty string.",
        args_schema=_ChartPayloadInput,
    ),
    "extract_shareholding_pattern": shareholding_pattern_tool,
    "extract_fundamentals_header": fundamentals_header_tool,
    "price_movement_analysis": price_movement_tool,
}


# ---------- Tool categories ----------
TOOL_CATEGORIES: Dict[str, List[str]] = {
    "web_search": ["exa_search", "exa_live_search", "fetch_url_text"],
    "chart": ["emit_chart"],
    "financial_analysis": ["extract_shareholding_pattern", "extract_fundamentals_header", "price_movement_analysis"],
}


def load_tools(use_cases: List[str] | None = None, structured: bool = False):
    """Return a list of tools based on use-cases. If structured=True, return StructuredTools.

    use_cases examples: ["web_search"], ["similarity"], ["web_search", "similarity"].
    If None or empty, returns all tools.
    """
    name_to_tool = STRUCTURED_TOOLS if structured else ALL_TOOLS
    print(
        f"[Registry] load_tools called | use_cases={use_cases}, structured={structured}"
    )

    if not use_cases:
        tools = [name_to_tool[n] for n in name_to_tool.keys()]
        print(
            f"[Registry] load_tools selected all tools: {[getattr(t, 'name', str(t)) for t in tools]}"
        )
        return tools

    selected: List[str] = []
    for uc in use_cases:
        names = TOOL_CATEGORIES.get(uc, [])
        selected.extend(names)
    # Preserve order and uniqueness based on ALL_TOOLS order
    seen = set()
    ordered = []
    for n in name_to_tool.keys():
        if n in selected and n not in seen and n in name_to_tool:
            ordered.append(name_to_tool[n])
            seen.add(n)
    print(
        f"[Registry] load_tools selected: {[getattr(t, 'name', str(t)) for t in ordered]}"
    )
    return ordered
