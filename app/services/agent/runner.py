from langchain_core.messages import HumanMessage, SystemMessage
from starlette.responses import StreamingResponse
import json
import re
import ast

from .builder import build_agent
from .memory import get_context
from app.stream_utils import normalize_stream_event, set_process_emitter


def _parse_json_or_python(s: str) -> dict | list | None:
    """Parse as strict JSON first; if it fails, parse as Python literal (single quotes)."""
    try:
        obj = json.loads(s)
        if isinstance(obj, (dict, list)):
            return obj
    except Exception:
        pass
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, (dict, list)):
            return obj
    except Exception:
        pass
    return None


def _extract_first_structured(text: str) -> dict | list | None:
    """Return the first dict/list found, accepting JSON or Python-literal style."""
    if not isinstance(text, str):
        return None
    s = text.strip()
    # Whole string
    obj = _parse_json_or_python(s)
    if obj is not None:
        return obj
    # Fenced blocks
    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", s, flags=re.IGNORECASE)
    for block in fenced:
        candidate = block.strip()
        if not candidate:
            continue
        obj = _parse_json_or_python(candidate)
        if obj is not None:
            return obj
    # Scan for first braced substring; try JSON then python literal
    for i, ch in enumerate(s):
        if ch in '{[':
            # Try JSON raw decode window
            try:
                dec = json.JSONDecoder()
                _, end = dec.raw_decode(s, i)
                obj = _parse_json_or_python(s[i:end])
                if obj is not None:
                    return obj
            except Exception:
                # Find matching close with a simple stack and try parse
                stack = [ch]
                for j in range(i + 1, len(s)):
                    cj = s[j]
                    if cj in '{[':
                        stack.append(cj)
                    elif cj in '}]':
                        if not stack:
                            break
                        top = stack.pop()
                        if (top == '{' and cj != '}') or (top == '[' and cj != ']'):
                            # mismatched; continue
                            stack.append(top)  # put back; keep searching
                    if not stack:
                        segment = s[i:j + 1]
                        obj = _parse_json_or_python(segment)
                        if obj is not None:
                            return obj
                        break
            # continue scanning if not parsed
    return None


# --- helper to remap all charts ---
def _remap_charts(parsed: dict) -> list[dict]:
    """
    Normalize chart payloads into [{type, data:{labels, values}, title}].
    Supports either:
    - direct chart_data: { event: 'chart_data', charts: [...] }
    - legacy: { chartable: true, aiChartData: [...] }
    """
    charts: list[dict] = []
    try:
        # Direct chart_data shape
        if isinstance(parsed, dict) and parsed.get("event") == "chart_data" and isinstance(parsed.get("charts"), list):
            for c in parsed["charts"]:
                if not isinstance(c, dict):
                    continue
                data = c.get("data") or {}
                charts.append({
                    "type": c.get("type"),
                    "data": {
                        "labels": (data.get("labels") if isinstance(data, dict) else []) or [],
                        "values": (data.get("values") if isinstance(data, dict) else []) or [],
                    },
                    "title": c.get("title", ""),
                })
        # Legacy aiChartData shape
        elif parsed.get("chartable") and parsed.get("aiChartData"):
            for ai_chart in parsed["aiChartData"]:
                chart_type = (ai_chart.get("type") or "").replace("-standard", "")
                labels, values = [], []
                for d in ai_chart.get("data", []):
                    label_key = ai_chart.get("xAxisKey") or ai_chart.get("nameKey")
                    value_key = ai_chart.get("dataKey")
                    if label_key and value_key:
                        labels.append(d.get(label_key))
                        values.append(d.get(value_key))
                charts.append({
                    "type": chart_type,
                    "data": {"labels": labels, "values": values},
                    "title": ai_chart.get("title", ""),
                })
    except Exception:
        pass
    return charts


# --- helper: unwrap {"answer": {...}} if present ---
def _unwrap_answer_container(obj: dict | list) -> dict | list:
    if isinstance(obj, dict) and "answer" in obj and isinstance(obj["answer"], (dict, list)):
        return obj["answer"]
    return obj


# System instruction for hybrid Chat + Chart via LangGraph tools
SYSTEM_SPEC = (
    "You can chat and you can generate charts.\n"
    "If the user asks for a chart (graph/plot/visualize), call the 'chart_generate' tool.\n"
    "If you need facts, briefly use EXA tools first, then pass a short 'supporting_input' to 'chart_generate'.\n"
    "When a chart is requested, your FINAL message must be ONLY this JSON object (no markdown/code fences, not wrapped in quotes, use double quotes):\n"
    "{ \"answer\": { \"event\": \"chart_data\", \"charts\": [ { \"type\": \"pie|bar|line|area|scatter\", \"data\": { \"labels\": string[], \"values\": number[] }, \"title\": string } ] } }"
)


# -------- Non-streaming agent response (all charts) --------
def agent_answer(question: str, user_id: str | None = None, chat_id: str | None = None):
    agent = build_agent(use_cases=["web_search", "similarity", "charts"], structured=True)
    system_msg = SystemMessage(content=SYSTEM_SPEC)
    user_msg = HumanMessage(content=question)
    context_msgs = get_context(chat_id, question) if chat_id else []
    state = {"messages": context_msgs + [system_msg, user_msg]}

    result = agent.invoke(state)
    messages = result.get("messages", [])

    if messages:
        raw = messages[-1].content
        # Try to detect chart JSON in the final message first (forgiving of single quotes)
        if isinstance(raw, str):
            parsed = _extract_first_structured(raw)
            if parsed is not None:
                try:
                    parsed = _unwrap_answer_container(parsed)
                    charts = _remap_charts(parsed)
                    if charts:
                        return {"event": "chart_data", "charts": charts}
                except Exception:
                    pass
            # If not chart JSON, treat as chat text
            return raw
        elif isinstance(raw, (dict, list)):
            parsed = _unwrap_answer_container(raw)
            charts = _remap_charts(parsed if isinstance(parsed, dict) else {})
            if charts:
                return {"event": "chart_data", "charts": charts}
    # Finally, return empty string (no chart, no text)
    return ""


# -------- Streaming agent response --------
async def agent_stream_response(question: str, user_id: str | None = None, chat_id: str | None = None):
    agent = build_agent(use_cases=["web_search", "similarity", "charts"], structured=True)
    system = SystemMessage(content=SYSTEM_SPEC)
    user = HumanMessage(content=question)
    context_msgs = get_context(chat_id, question) if chat_id else []

    async def token_generator():
        index = 0
        _pending_meta: list[bytes] = []
        emitted_chart = False

        def _emitter(obj):
            payload = obj if isinstance(obj, dict) else {"event": "process", "message": str(obj)}
            _pending_meta.append(f"data: {json.dumps(payload)}\n\n".encode("utf-8"))

        set_process_emitter(_emitter)
        if chat_id:
            _pending_meta.append(f"data: {json.dumps({'event': 'meta', 'chat_id': chat_id})}\n\n".encode("utf-8"))

        try:
            if hasattr(agent, "astream_events"):
                async for event in agent.astream_events({"messages": context_msgs + [system, user]}, version="v1"):
                    while _pending_meta:
                        yield _pending_meta.pop(0)

                    text = normalize_stream_event(event)
                    if text:
                        yield f"data: {json.dumps({'event': 'token', 'text': text, 'index': index})}\n\n".encode("utf-8")
                        index += 1

                    # detect chart tool output
                    if isinstance(event, dict) and event.get("event") == "on_tool_end":
                        data = event.get("data")
                        if data:
                            try:
                                # If tool returns dict/list, use directly; else try to extract JSON (forgiving)
                                if isinstance(data, (dict, list)):
                                    parsed = _unwrap_answer_container(data)
                                else:
                                    parsed = _extract_first_structured(str(data))
                                    if parsed is None:
                                        continue
                                    parsed = _unwrap_answer_container(parsed)
                                charts = _remap_charts(parsed)
                                if charts:
                                    # STREAM ALL CHARTS
                                    yield f"data: {json.dumps({'event': 'chart_data', 'charts': charts})}\n\n".encode("utf-8")
                                    emitted_chart = True
                            except Exception:
                                pass
            else:
                async for event in agent.astream({"messages": context_msgs + [system, user]}, stream_mode="values"):
                    while _pending_meta:
                        yield _pending_meta.pop(0)
                    msgs = event.get("messages") if isinstance(event, dict) else None
                    if not msgs:
                        continue
                    last = msgs[-1]
                    if getattr(last, "type", "") != "ai":
                        continue
                    text = getattr(last, "content", None) or ""
                    if not text:
                        continue
                    # Stream raw text; if the model happened to include JSON, we don't force it
                    # into JSON-only mode anymore since chat is allowed.
                    for i in range(0, len(text), 400):
                        piece = text[i:i+400]
                        yield f"data: {json.dumps({'event': 'token', 'text': piece, 'index': index})}\n\n".encode("utf-8")
                        index += 1
        finally:
            yield f"data: {json.dumps({'event': 'end'})}\n\n".encode("utf-8")

    headers = {
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(token_generator(), media_type="text/event-stream", headers=headers)
