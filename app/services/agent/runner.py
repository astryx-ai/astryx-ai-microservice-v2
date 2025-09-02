from langchain_core.messages import HumanMessage, SystemMessage
from starlette.responses import StreamingResponse
import json
import re

from .builder import build_agent
from .memory import get_context
from app.stream_utils import normalize_stream_event, set_process_emitter


# --- helper to remove Markdown code blocks if present ---
def _strip_markdown_json(text: str) -> str:
    """
    Remove Markdown code blocks like ```json ... ``` or ``` ... ``` if present.
    """
    match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    return text


# --- helper to remap all charts ---
def _remap_charts(parsed: dict) -> list[dict]:
    """
    Convert all charts in aiChartData JSON into compact {type, data:{labels, values}, title}.
    Returns a list of charts.
    """
    charts = []
    try:
        if parsed.get("chartable") and parsed.get("aiChartData"):
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


# -------- Non-streaming agent response (all charts) --------
def agent_answer(question: str, user_id: str | None = None, chat_id: str | None = None):
    agent = build_agent(use_cases=["web_search", "similarity", "charts"], structured=True)
    system_msg = SystemMessage(content="Return chart data ONLY as raw JSON. No Markdown or prose.")
    user_msg = HumanMessage(content=question)
    context_msgs = get_context(chat_id, question) if chat_id else []
    state = {"messages": context_msgs + [system_msg, user_msg]}

    result = agent.invoke(state)
    messages = result.get("messages", [])

    if messages:
        last_content = _strip_markdown_json(messages[-1].content)
        try:
            parsed = json.loads(last_content)
            charts = _remap_charts(parsed)
            if charts:
                # RETURN ALL CHARTS exactly like streaming version
                return {"event": "chart_data", "charts": charts}
        except Exception:
            return last_content
    return {}


# -------- Streaming agent response --------
async def agent_stream_response(question: str, user_id: str | None = None, chat_id: str | None = None):
    agent = build_agent(use_cases=["web_search", "similarity", "charts"], structured=True)
    system = SystemMessage(content="Return chart data ONLY as raw JSON. No Markdown or prose.")
    user = HumanMessage(content=question)
    context_msgs = get_context(chat_id, question) if chat_id else []

    async def token_generator():
        index = 0
        _pending_meta: list[bytes] = []

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
                                data_str = _strip_markdown_json(str(data))
                                parsed = json.loads(data_str)
                                charts = _remap_charts(parsed)
                                if charts:
                                    # STREAM ALL CHARTS
                                    yield f"data: {json.dumps({'event': 'chart_data', 'charts': charts})}\n\n".encode("utf-8")
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
                    text = _strip_markdown_json(text)
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
