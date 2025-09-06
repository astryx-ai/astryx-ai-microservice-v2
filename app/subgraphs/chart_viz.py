from langgraph.prebuilt import create_react_agent
from langchain.tools import StructuredTool  # noqa: F401 (kept for historical context if needed)
from pydantic import BaseModel, Field
from app.services.llms.azure_openai import chat_model
from app.utils.stream_utils import emit_process
from app.agent_tools.registry import load_tools
from app.agent_tools.formatter import format_financial_content


class ExaSearchInput(BaseModel):
    query: str = Field(..., description="Search query")
    max_results: int = Field(5, ge=1, le=20, description="Maximum number of results")


class ExaLiveSearchInput(BaseModel):
    query: str = Field(..., description="Live search query")
    k: int = Field(8, ge=1, le=20, description="Number of documents to retrieve")
    max_chars: int = Field(1000, ge=100, le=8000, description="Maximum characters in each summary")


class FetchUrlTextInput(BaseModel):
    url: str = Field(..., description="Web page URL to fetch in raw text chunks")
    chunk_index: int = Field(1, ge=1, le=1000, description="1-based index of chunk to return")
    chunk_size: int = Field(4000, ge=500, le=20000, description="Approximate characters per chunk")
    max_total_chars: int = Field(120000, ge=5000, le=500000, description="Safety cap on total extracted characters")


def _get_chart_viz_tools():
    return load_tools(use_cases=["web_search", "chart"], structured=True)


def run_chart_viz(query: str, context_messages=None) -> str:
    print(f"[ChartViz] Starting chart visualization | query='{query}'")
    try:
        llm = chat_model(temperature=0.1)
        tools = _get_chart_viz_tools()
        agent = create_react_agent(llm, tools)
        conversation_context = ""
        if context_messages:
            print(f"[ChartViz] Including {len(context_messages)} context messages")
            context_parts = []
            for msg in context_messages[-6:]:
                if hasattr(msg, 'type'):
                    role = "User" if msg.type == "human" else "Assistant"
                    content = getattr(msg, 'content', str(msg))
                    if content and not content.startswith("You are"):
                        context_parts.append(f"{role}: {content[:500]}")
            if context_parts:
                conversation_context = "\n\nCONVERSATION CONTEXT:\n" + "\n".join(context_parts[-4:])
        system = (
            "You are a data collection and analysis agent for charting. "
            "Your job is to search and extract structured numerical facts (labels and values) from the web. "
            "DO NOT include any JSON in your text output. "
            "When you have enough cleaned data points for a chart, CALL the tool 'emit_chart' with one argument named 'payload' "
            "containing the full chart JSON. You MAY call 'emit_chart' multiple times within the same run to emit multiple different charts. "
            "Choose the chart 'type' that best fits the data (e.g., line-linear for time series, bar-standard for single series, bar-multiple/stacked for grouped data, pie for composition). "
            "Continue your analysis after calling the tool; do not repeat any JSON in text. "
            "If needed, use 'fetch_url_text' to gather detailed content.\n\n"
            "WRITING STYLE:\n"
            "- Begin with a 1–2 sentence 'Chart insight' lead that explains the overall trend or takeaway.\n"
            "- After the lead, insert a blank line, then use '##' markdown headings for sections.\n"
            "- Ensure proper markdown hygiene: never place a heading immediately after a sentence without a blank line.\n\n"
            "SUPPORTED CHART TYPES AND REQUIRED KEYS (must be present in payload):\n"
            "- bar-standard: [type, title, description, dataKey, nameKey, data]\n"
            "- bar-multiple: [type, title, description, dataKey, nameKey, xAxisKey, layout, data]\n"
            "- bar-stacked: [type, title, description, xAxisKey, nameKey, groupedKeys, data]\n"
            "- bar-negative: [type, title, description, xAxisKey, groupedKeys, legend, data]\n"
            "- area-standard: [type, title, description, dataKey, nameKey, color, data]\n"
            "- area-linear: [type, title, description, dataKey, nameKey, color, data]\n"
            "- area-stacked: [type, title, description, dataKey, groupedKeys, color, legend, data]\n"
            "- area-stacked-alt: [type, title, description, groupedKeys, xAxisKey, data]\n"
            "- line-standard: [type, title, description, dataKey, nameKey, color, data]\n"
            "- line-linear: [type, title, description, dataKey, nameKey, color, legend, data]\n"
            "- line-multiple: [type, title, description, dataKey, groupedKeys, color, legend, data]\n"
            "- line-dots: [type, title, description, dataKey, color, legend, dots, data]\n"
            "- line-label: [type, title, description, dataKey, xAxisKey, customLabel, data]\n"
            "- line-label-2: [type, title, description, dataKey, nameKey, color, label, data]\n"
            "- pie-standard: [type, title, description, dataKey, nameKey, data]\n"
            "- pie-label: [type, title, description, dataKey, nameKey, label, legend, data]\n"
            "- pie-interactive: [type, title, description, dataKey, nameKey, legend, data]\n"
            "- pie-donut: [type, title, description, dataKey, nameKey, legend, data]\n"
            "- radar-standard: [type, title, description, dataKey, nameKey, color, legend, data]\n"
            "- radar-lines-only: [type, title, description, dataKey, nameKey, color, legend, data]\n"
            "- radar-multiple: [type, title, description, dataKey, color, legend, grid, data]\n"
            "- radial-standard: [type, title, description, dataKey, groupedKeys, color, legend, data]\n"
            "- radial-stacked: [type, title, description, groupedKeys, nameKey, data]\n"
            "- radial-progress: [type, title, description, dataKey, nameKey, centerText, centerLabel, data]\n\n"
            "DEFAULTS: If a required key is not obvious, include a sensible default: "
            "color='#4e79a7'; legend=['Series'] or []; layout='vertical'; grid=true; dots=true; "
            "customLabel='value'; label='value'."
            f"{conversation_context}"
        )
        user = (
            f"Task: {query}\n"
            f"Instructions:\n"
            f"- Search the web for relevant, recent numerical data points.\n"
            f"- Extract and normalize them into label/value pairs.\n"
            f"- Decide the most appropriate chart type (bar/line/area/pie/radar/radial variants).\n"
            f"- Build a VALID payload including ALL required keys for the chosen type (see list above).\n"
            f"- Example for time series: line-linear requires [type,title,description,dataKey,nameKey,color,legend,data].\n"
            f"- Then CALL tool emit_chart with one argument named payload containing the full JSON. If the user requested multiple charts or multiple views are clearly helpful, call emit_chart MULTIPLE times—once per chart.\n"
            f"- DO NOT include any JSON in the textual response. Only use the tool to emit the chart.\n"
            f"- After calling the tool, continue with a succinct analysis in prose only.\n"
            f"- Start your response with a 'Chart insight' lead (1–2 sentences), then a blank line, then '##' headings."
        )
        try:
            emit_process({"message": "Chart Visualization started"})
            emit_process({"message": "Generating chart"})
        except Exception:
            pass
        result = agent.invoke({"messages": [
            {"type": "system", "content": system},
            {"type": "human", "content": user},
        ]})
        print(f"[ChartViz] Agent invoke result type: {type(result)}")
        if isinstance(result, dict):
            messages = result.get("messages", [])
            if messages:
                last_message = messages[-1]
                content = getattr(last_message, "content", "") if hasattr(last_message, "content") else str(last_message)
            else:
                content = ""
        elif isinstance(result, str):
            content = result
        else:
            content = str(result)
        try:
            emit_process({"message": "Chart Visualization completed"})
        except Exception:
            pass
        try:
            cleaned = format_financial_content(content or "")
        except Exception:
            cleaned = content or ""
        return cleaned
    except Exception as e:
        return f"Chart visualization failed: {e}"


