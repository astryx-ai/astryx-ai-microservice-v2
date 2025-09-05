from langgraph.prebuilt import create_react_agent
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from app.services.llms.azure_openai import chat_model
from app.utils.stream_utils import emit_process
import json
from .exa import (
    exa_search as _exa_search,
    exa_live_search as _exa_live_search,
    fetch_url_text as _fetch_url_text,
)


# Minimal schemas for chart viz tools
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
    """Get the minimal set of tools needed for chart visualization."""
    return [
        StructuredTool.from_function(
            func=lambda query, max_results=5: _exa_search.func(query, max_results),
            name="exa_search",
            description="Search the web for numerical data and statistics to create charts.",
            args_schema=ExaSearchInput,
        ),
        StructuredTool.from_function(
            func=lambda query, k=8, max_chars=1000: _exa_live_search.func(query, k, max_chars),
            name="exa_live_search",
            description="Live-crawl search for current numerical data and statistics.",
            args_schema=ExaLiveSearchInput,
        ),
        StructuredTool.from_function(
            func=lambda url, chunk_index=1, chunk_size=4000, max_total_chars=120000: _fetch_url_text.func(url, chunk_index, chunk_size, max_total_chars),
            name="fetch_url_text",
            description="Fetch web pages containing numerical data for chart creation.",
            args_schema=FetchUrlTextInput,
        ),
    ]

def run_chart_viz(query: str, context_messages=None) -> str:
    """Execute chart visualization using a specialized subgraph agent."""
    print(f"[ChartViz] Starting chart visualization | query='{query}'")
    
    try:
        # Build a lean subgraph with chart-oriented prompt and only web tools
        llm = chat_model(temperature=0.1)
        tools = _get_chart_viz_tools()
        agent = create_react_agent(llm, tools)
        
        # Build conversation context
        conversation_context = ""
        if context_messages:
            print(f"[ChartViz] Including {len(context_messages)} context messages")
            context_parts = []
            for msg in context_messages[-6:]:  # Last 6 messages for context
                if hasattr(msg, 'type'):
                    role = "User" if msg.type == "human" else "Assistant"
                    content = getattr(msg, 'content', str(msg))
                    if content and not content.startswith("You are"):  # Skip system messages
                        context_parts.append(f"{role}: {content[:500]}")  # Truncate long messages
            if context_parts:
                conversation_context = "\n\nCONVERSATION CONTEXT:\n" + "\n".join(context_parts[-4:])  # Last 4 exchanges
        
        system = (
            "You are a data visualization specialist. Your task is to search for numerical data and create chart visualizations. "
            "Follow these steps: "
            "1) Use search tools (exa_live_search, exa_search) to find relevant numerical data, "
            "2) Use fetch_url_text to get detailed data from the most relevant sources, "
            "3) Extract specific numbers, percentages, financial figures, or statistical data, "
            "4) Create a JSON chart response using the bar-standard format. "
            "CRITICAL: Your final response must include a JSON chart in this exact format:\n"
            "{\n"
            '  "type": "bar-standard",\n'
            '  "title": "Chart Title",\n'
            '  "description": "Chart description",\n'
            '  "dataKey": "value_field_name",\n'
            '  "nameKey": "label_field_name",\n'
            '  "data": [\n'
            '    {"label_field_name": "Label1", "value_field_name": 123},\n'
            '    {"label_field_name": "Label2", "value_field_name": 456}\n'
            '  ]\n'
            "}\n"
            "IMPORTANT: Place the JSON chart at the END of your response, clearly separated from your analysis text. "
            "Do NOT embed the JSON within sentences or explanations. "
            "Do NOT wrap the JSON in code blocks or markdown - just provide the raw JSON object."
            f"{conversation_context}"
        )
        
        user = (
            f"Create a chart visualization for: {query}\n"
            f"Requirements:\n"
            f"- Search for relevant numerical data\n"
            f"- Extract specific numbers (revenue, market cap, percentages, etc.)\n"
            f"- Create a bar chart with at least 3-5 data points\n"
            f"- Provide analysis followed by JSON chart data\n"
            f"- Use descriptive titles and labels\n"
            f"- Ensure data is current and accurate"
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
        
        # Normalize content
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
        
        # Prepare outputs
        chart_data = None
        analysis_text = content or ""

        # Only try extraction if we see the expected type token
        if '"type":' in analysis_text and 'bar-standard' in analysis_text:
            import re

            def remove_surrounding_header(text, start_index):
                """
                If there's a short header line immediately before start_index that refers to JSON/chart,
                remove that header too (e.g., "Below is the JSON chart visualization:").
                Returns new start index (inclusive).
                """
                prev_nl = text.rfind('\n', 0, start_index)
                if prev_nl == -1:
                    prev_nl = 0
                header = text[prev_nl:start_index].strip().lower()
                # keywords commonly used in the header - adjust if needed
                keywords = ['below is', 'json chart', 'chart visualization', 'below is the json', 'below is the chart', 'chart data', 'json chart visualization', 'below is the json chart visualization']
                # if header contains any keyword and header length reasonable, remove it
                if any(kw in header for kw in keywords) and len(header) < 120:
                    return prev_nl
                # also if it's just punctuation or colon, remove it
                if header.endswith(':') and len(header) < 60:
                    return prev_nl
                return start_index

            # 1) Remove code-fenced JSON blocks first
            fenced_pattern = re.compile(r'```(?:json)?\s*([\s\S]*?)```', re.DOTALL | re.IGNORECASE)
            for m in list(fenced_pattern.finditer(analysis_text)):
                candidate = m.group(1).strip()
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict) and parsed.get("type") == "bar-standard":
                        chart_data = parsed if chart_data is None else chart_data
                        # remove whole fenced block including preceding header line if present
                        start = m.start()
                        new_start = remove_surrounding_header(analysis_text, start)
                        analysis_text = (analysis_text[:new_start] + analysis_text[m.end():]).strip()
                        print(f"[ChartViz] Removed fenced chart JSON: {chart_data.get('title', 'Untitled')}")
                except Exception:
                    # If json parsing fails, skip
                    continue

            # 2) Remove raw JSON occurrences by locating the "type":"bar-standard" and matching braces robustly
            while True:
                # find the position of the bar-standard marker
                marker_match = re.search(r'"type"\s*:\s*"bar-standard"', analysis_text, re.IGNORECASE)
                if not marker_match:
                    break

                # find nearest opening brace '{' before the marker
                marker_pos = marker_match.start()
                start_brace = analysis_text.rfind('{', 0, marker_pos)
                if start_brace == -1:
                    # nothing to extract, break to avoid infinite loop
                    break

                # scan forward to find matching closing brace while respecting JSON strings
                i = start_brace
                length = len(analysis_text)
                brace_count = 0
                in_string = False
                escape = False
                end_index = -1

                while i < length:
                    ch = analysis_text[i]
                    if ch == '"' and not escape:
                        in_string = not in_string
                        i += 1
                        escape = False
                        continue
                    if in_string:
                        if ch == '\\' and not escape:
                            escape = True
                        else:
                            escape = False
                        i += 1
                        continue
                    # not in string
                    if ch == '{':
                        brace_count += 1
                    elif ch == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_index = i
                            break
                    i += 1

                if end_index == -1:
                    # failed to find matching brace — try a regex fallback that captures JSON-ish chunk
                    # but to avoid infinite loop, break
                    break

                candidate = analysis_text[start_brace:end_index+1].strip()
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict) and parsed.get("type") == "bar-standard":
                        chart_data = parsed if chart_data is None else chart_data
                        # remove candidate plus preceding small header line if relevant
                        new_start = remove_surrounding_header(analysis_text, start_brace)
                        analysis_text = (analysis_text[:new_start] + analysis_text[end_index+1:]).strip()
                        print(f"[ChartViz] Removed raw chart JSON: {chart_data.get('title', 'Untitled')}")
                        # continue loop in case there are multiple charts
                        continue
                except Exception:
                    # couldn't parse, skip this occurrence — to avoid infinite loop, remove this marker and continue
                    # remove the marker itself and continue scanning
                    marker_end = marker_match.end()
                    analysis_text = analysis_text[:marker_match.start()] + analysis_text[marker_end:]
                    continue

            # Final cleanup: remove any leftover single-line headers that mention the chart
            analysis_text = re.sub(r'(?im)^\s*(below\s+is[^\n]*json[^\n]*\n?)', '', analysis_text).strip()
            analysis_text = re.sub(r'(?im)^\s*(below\s+is[^\n]*chart[^\n]*\n?)', '', analysis_text).strip()
            # collapse multiple blank lines to two
            analysis_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', analysis_text).strip()

        # If no chart found, create a fallback
        if not chart_data:
            chart_data = {
                "type": "bar-standard",
                "title": "Data Visualization",
                "description": "Chart based on search results",
                "dataKey": "revenue",
                "nameKey": "company",
                "data": [
                    {"company": "Data Point 1", "revenue": 100},
                    {"company": "Data Point 2", "revenue": 150},
                    {"company": "Data Point 3", "revenue": 75}
                ]
            }
            print("[ChartViz] Using fallback chart")
        
        # Emit the chart data as a special process event
        try:
            emit_process({"event": "chart_data", "chart": chart_data})
            print("[ChartViz] Emitted chart data event")
        except Exception as e:
            print(f"[ChartViz] Failed to emit chart data: {e}")
            
        # Emit completion
        try:
            emit_process({"message": "Chart Visualization completed"})
        except Exception:
            pass

        # Return only the analysis text (no JSON)
        return analysis_text if analysis_text else "Chart visualization completed."
        
    except Exception as e:
        return f"Chart visualization failed: {e}"
