from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv, find_dotenv

from app.config import settings
from app.stream_utils import emit_process


def _load_chart_knowledge() -> str:
    """Load chart knowledge base from chart.json next to this module."""
    try:
        chart_json_path = Path(__file__).resolve().parent / "chart.json"
        data = chart_json_path.read_text(encoding="utf-8")
        return data
    except Exception:
        return "{}"


def generate_graph(query: str, supporting_input: Optional[str] = None, chart_type_hint: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate chart JSON for any chart type (pie, line, bar, area, radar)
    """
    emit_process({"message": "Generating chart configuration"})

    # Load chart knowledge once
    chart_knowledge = _load_chart_knowledge()

    system_prompt = (
        "You are a data visualization expert. "
        "Given a user's query and optional supporting input, "
        "determine if it can be represented as a chart and generate valid JSON ONLY.\n\n"
        "JSON format:\n"
        "{\n"
        '  "chartable": true,\n'
        '  "aiChartData": [\n'
        "    {\n"
        '      "data": [...],\n'
        '      "type": "pie-standard / line-standard / bar-standard / area-standard / radar-standard",\n'
        '      "title": "Chart title",\n'
        '      "dataKey": "numerical value key",\n'
        '      "nameKey": "label key for pie charts",\n'
        '      "xAxisKey": "x-axis key for line/bar charts",\n'
        '      "description": "Brief description"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "If data is not suitable for charting, return:\n"
        "{\n"
        '  "chartable": false,\n'
        '  "reason": "Explain why"\n'
        "}\n"
        f"Supporting chart knowledge: {chart_knowledge}\n"
    )

    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Query: {query}\nSupporting Input: {supporting_input or 'None'}"),
        ]

        graph_llm = AzureChatOpenAI(
            azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT or os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT or os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_key=settings.AZURE_OPENAI_API_KEY or os.getenv("AZURE_OPENAI_API_KEY"),
            openai_api_version=settings.AZURE_OPENAI_API_VERSION or os.getenv("AZURE_OPENAI_API_VERSION"),
            temperature=0.1,
            model_kwargs={"response_format": {"type": "json_object"}},
            streaming=False,
        )

        result = graph_llm.invoke(messages)
        content = getattr(result, "content", str(result))

    except Exception as e:
        return {"chartable": False, "reason": f"Failed to invoke LLM: {e}"}

    # Safe JSON parsing
    try:
        parsed = json.loads(content)
        if parsed.get("chartable") and parsed.get("aiChartData") and chart_type_hint:
            # Enforce chart type hint on first chart
            parsed["aiChartData"][0]["type"] = chart_type_hint
        return parsed
    except json.JSONDecodeError:
        return {"chartable": False, "reason": "Invalid JSON from LLM"}


# -------- Tool-facing wrappers --------
def chart_generate(query: str, supporting_input: Optional[str] = None) -> str:
    """Structured tool entrypoint. Returns a JSON string of the chart config/result."""
    emit_process({"message": "chart_generate tool running"})
    out = generate_graph(query=query, supporting_input=supporting_input)
    try:
        return json.dumps(out)
    except Exception:
        return json.dumps({"chartable": False, "reason": "Serialization error"})


def run(state: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Direct callable used by runner for fast chart-only path.

    Expects state like {"query": str, "supporting_input": Optional[str], "chart_type": Optional[str]}.
    Returns {"chart_event": <parsed-json>} for streaming via /agent/stream.
    """
    state = state or {}
    query = state.get("query") or ""
    supporting_input = state.get("supporting_input")
    result = generate_graph(query=query, supporting_input=supporting_input)

    # If caller hinted a chart_type, try to enforce it on first dataset when chartable
    hint = state.get("chart_type")
    if hint and isinstance(result, dict) and result.get("chartable") and isinstance(result.get("aiChartData"), list) and result["aiChartData"]:
        try:
            result["aiChartData"][0]["type"] = hint
        except Exception:
            pass

    return {"chart_event": result}


# Export for registry convenience
chart_run = run


# Structured tool declaration
try:
    from pydantic import BaseModel, Field
    from langchain.tools import StructuredTool

    class ChartGenerateInput(BaseModel):
        query: str = Field(..., description="User query to visualize")
        supporting_input: Optional[str] = Field(None, description="Optional supporting data or context")

    CHART_GENERATE_TOOL = StructuredTool.from_function(
        func=chart_generate,
        name="chart_generate",
        description=(
            "Generate an Astryx-compatible chart JSON from a query and optional supporting input. "
            "Always use when the user asks for a chart."
        ),
        args_schema=ChartGenerateInput,
    )
except Exception:
    CHART_GENERATE_TOOL = None
