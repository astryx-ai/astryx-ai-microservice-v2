from langchain_core.messages import HumanMessage, SystemMessage
from starlette.responses import StreamingResponse
import json

from .builder import build_agent
from .memory import get_context
from app.stream_utils import normalize_stream_event, set_process_emitter

# Agent responses without streaming
def agent_answer(question: str, user_id: str | None = None, chat_id: str | None = None) -> str:
    """Run the LangGraph ReAct agent with EXA tools on a single question and return the final answer."""
    graph = build_agent(use_cases=["web_search", "similarity"], structured=True)
    system_msg = SystemMessage(content=(
        "You can search the web using the provided EXA tools when helpful. "
        "Prefer up-to-date sources. Be concise and cite sources inline."
    ))
    user_msg = HumanMessage(content=(
        f"Task: {question}\n"
        "If web context is needed: 1) prefer exa_live_search for fresh results, otherwise exa_search; "
        "2) optionally use fetch_url on top 1-3 URLs to extract short summaries, "
        "3) answer concisely with bullet points and cite links inline."
    ))
    context_msgs = get_context(chat_id, question) if chat_id else []
    state = {"messages": context_msgs + [system_msg, user_msg]}
    result = graph.invoke(state)
    messages = result.get("messages", [])
    return messages[-1].content if messages else ""


# Agent responses with streaming
async def agent_stream_response(question: str, user_id: str | None = None, chat_id: str | None = None):
    agent = build_agent(use_cases=["web_search", "similarity"], structured=True)
    system = SystemMessage(content=(
        "You can search the web using EXA tools when needed. Prefer up-to-date sources."
    ))
    user = HumanMessage(content=question)

    context_msgs = get_context(chat_id, question) if chat_id else []

    async def token_generator():
        index = 0
        sent_any = False

        def _emit_process_line(obj):
            nonlocal index
            payload = {"process": obj.get("message", ""), **{k: v for k, v in obj.items() if k != "message"}}
            _pending_meta.append((json.dumps({"meta": payload}) + "\n").encode("utf-8"))

        try:
            # Install process emitter into context so memory/tools can publish
            set_process_emitter(lambda obj: (_ for _ in ()).throw(StopIteration))  # placeholder in case called before loop
            if hasattr(agent, "astream_events"):
                # Rebind to emit via outer generator "yield" using event schema
                def _emitter(obj):
                    nonlocal index
                    payload = obj if isinstance(obj, dict) else {"event": "process", "message": str(obj)}
                    raise_yield = (json.dumps(payload) + "\n").encode("utf-8")
                    nonlocal _pending_meta
                    _pending_meta.append(raise_yield)

                _pending_meta: list[bytes] = []
                set_process_emitter(_emitter)
                # Emit an initial meta with chat_id if present
                if chat_id:
                    _pending_meta.append((json.dumps({"meta": {"chat_id": chat_id}}) + "\n").encode("utf-8"))
                async for event in agent.astream_events({"messages": context_msgs + [system, user]}, version="v1"):
                    while _pending_meta:
                        yield _pending_meta.pop(0)
                    text = normalize_stream_event(event)
                    if text:
                        sent_any = True
                        yield (json.dumps({"event": "token", "text": text, "index": index}) + "\n").encode("utf-8")
                        index += 1
            else:
                def _emitter(obj):
                    nonlocal index
                    payload = obj if isinstance(obj, dict) else {"event": "process", "message": str(obj)}
                    _pending_meta.append((json.dumps(payload) + "\n").encode("utf-8"))

                _pending_meta = []
                set_process_emitter(_emitter)
                if chat_id:
                    _pending_meta.append((json.dumps({"meta": {"chat_id": chat_id}}) + "\n").encode("utf-8"))
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
                    sent_any = True
                    chunk_size = 400
                    for i in range(0, len(text), chunk_size):
                        piece = text[i:i+chunk_size]
                        if piece:
                            yield (json.dumps({"event": "token", "text": piece, "index": index}) + "\n").encode("utf-8")
                            index += 1
        except Exception:
            sent_any = False
        if not sent_any:
            try:
                # ensure any pending meta is flushed before fallback content
                resp = agent.invoke({"messages": context_msgs + [system, user]})
                # flush pending meta
                try:
                    while _pending_meta:
                        yield _pending_meta.pop(0)
                except Exception:
                    pass
                content = getattr(resp.get("messages", [])[-1], "content", "") if isinstance(resp, dict) and resp.get("messages") else ""
                if content:
                    chunk_size = 400
                    for i in range(0, len(content), chunk_size):
                        piece = content[i:i+chunk_size]
                        if piece:
                            yield (json.dumps({"event": "token", "text": piece, "index": index}) + "\n").encode("utf-8")
                            index += 1
            except Exception:
                pass
        # Final end marker
        yield (json.dumps({"event": "end"}) + "\n").encode("utf-8")

    headers = {
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(token_generator(), media_type="text/plain", headers=headers)


