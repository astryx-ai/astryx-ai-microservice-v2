from langchain_core.messages import HumanMessage, SystemMessage
from starlette.responses import StreamingResponse
import json

from app.agent_tools.helper_tools import decide_route
from app.services.agent.state import AVAILABLE_ROUTES
from .builder import build_agent
from .memory import get_context
from app.utils.stream_utils import normalize_stream_event, set_process_emitter
from app.agent_tools.formatter import format_financial_content


"""
Note: Chart JSON extraction/processing removed. Chart emission now happens via chart tools
and process emitter inside chart_viz. Runner only formats and streams text output.
"""


# Agent responses without streaming
def agent_answer(question: str, user_id: str | None = None, chat_id: str | None = None) -> str:
    """Run the routed LangGraph agent and return the final answer."""
    print(f"[Runner] agent_answer invoked | user_id={user_id}, chat_id={chat_id}")

    # Build the routed graph
    graph = build_agent()

    # Simple system and user messages for the graph
    system_msg = SystemMessage(content=(
        "You are a financial AI assistant. "
        f"Current datetime (UTC): {get_current_datetime_string()}. "
        "Prefer the most up-to-date, reputable sources and verify recency when relevant. "
        "Use web tools without rewriting the user's query text; if recency matters, reason about the timeframe "
        "and reflect it in your plan/answer rather than modifying the tool query. "
        "Present results with clear structure, tables for numerical data, and proper citations."
    ))
    user_msg = HumanMessage(content=f"Task: {question}")

    # Get context and build state with pre-computed routing
    context_msgs = get_context(chat_id, question) if chat_id else []

    user_and_ai_messages = [msg for msg in context_msgs + [user_msg] if hasattr(msg, 'type') and msg.type in ["human", "ai"]]
    has_context = len(user_and_ai_messages) > 2

    print("[Runner] Pre-computing routing decision for non-streaming")
    try:
        available_routes = AVAILABLE_ROUTES
        route, reason = decide_route(question, has_context, context_messages=user_and_ai_messages, available_routes=available_routes)
        print(f"[Runner] Non-streaming route decision successful: {route}")
    except Exception as route_error:
        print(f"[Runner] Non-streaming route decision failed: {route_error}")
        route, reason = "standard", f"routing error: {route_error}"

    state = {
        "messages": context_msgs + [system_msg, user_msg],
        "route": route,
        "decision_reason": reason,
        "query": question,
        "context": user_and_ai_messages,
        "now_utc": get_current_datetime_string(),
    }
    print(f"[Runner] Pre-computed route: {route}")

    print("[Runner] agent_answer executing routed graph")
    # Capture sources via emitter during invoke
    sources: dict[str, str] = {}
    def _answer_emitter(obj):
        try:
            if isinstance(obj, dict) and obj.get("event") == "source":
                url = str(obj.get("url") or "").strip()
                title = str(obj.get("title") or url).strip()
                if url:
                    sources[url] = title
        except Exception:
            pass
    set_process_emitter(_answer_emitter)

    result = graph.invoke(state)
    print(f"[Runner] Non-streaming response type: {type(result)}")

    # Handle different response types
    if isinstance(result, dict):
        messages = result.get("messages", [])
        if messages:
            last_message = messages[-1]
            raw_content = getattr(last_message, "content", "") if hasattr(last_message, "content") else str(last_message)
        else:
            raw_content = ""
    elif isinstance(result, str):
        raw_content = result
    else:
        raw_content = str(result)

    # Format the content; chart emission is handled by tools inside chart_viz
    formatted_content = format_financial_content(raw_content)

    # Append citations if any sources captured
    if sources:
        citations_lines = ["", "## Citations"]
        for url, title in sources.items():
            citations_lines.append(f"- [{title}]({url})")
        formatted_content = formatted_content.rstrip() + "\n" + "\n".join(citations_lines) + "\n"

    return formatted_content


# Agent responses with streaming
async def agent_stream_response(question: str, user_id: str | None = None, chat_id: str | None = None):
    try:
        print(f"[Runner] agent_stream_response invoked | user_id={user_id}, chat_id={chat_id}")

        # Build the routed graph
        agent = build_agent()

        # Simple system and user messages for the graph
        system = SystemMessage(content=(
            "You are a financial AI assistant. "
            f"Current datetime (UTC): {get_current_datetime_string()}. "
            "Prefer the most up-to-date, reputable sources and verify recency when relevant. "
            "Use web tools without rewriting the user's query text; if recency matters, reason about the timeframe "
            "and reflect it in your plan/answer rather than modifying the tool query. "
            "Present results with clear structure, tables for numerical data, and proper citations."
        ))
        user = HumanMessage(content=f"Task: {question}")

        context_msgs = get_context(chat_id, question) if chat_id else []
    except Exception as setup_error:
        error_msg = f"Setup error: {str(setup_error)}"
        print(f"[Runner] {error_msg}")
        # Return a simple error response
        async def error_generator():
            yield (json.dumps({"event": "token", "text": error_msg, "index": 0}) + "\n").encode("utf-8")
            yield (json.dumps({"event": "end"}) + "\n").encode("utf-8")
        return StreamingResponse(error_generator(), media_type="text/plain")

    print(f"[Runner] Initial state setup - context_msgs: {len(context_msgs)}, system: {type(system)}, user: {type(user)}")

    # Pre-compute routing decision to avoid LLM call during streaming
    user_and_ai_messages = [msg for msg in context_msgs + [user] if hasattr(msg, 'type') and msg.type in ["human", "ai"]]
    has_context = len(user_and_ai_messages) > 2
    user_query = question  # Direct from the question parameter

    print("[Runner] Pre-computing routing decision to avoid streaming leaks")
    try:
        available_routes = AVAILABLE_ROUTES
        chosen_route, reason = decide_route(user_query, has_context, context_messages=user_and_ai_messages, available_routes=available_routes)
        print(f"[Runner] Route decision successful: {chosen_route}")
    except Exception as route_error:
        print(f"[Runner] Route decision failed: {route_error}")
        chosen_route, reason = "standard", f"routing error: {route_error}"

    # Pre-populate the state with routing decision
    initial_state = {
        "messages": context_msgs + [system, user],
        "route": chosen_route,
        "decision_reason": reason,
        "query": user_query,
        "now_utc": get_current_datetime_string(),
    }
    print(f"[Runner] Pre-computed route: {chosen_route}")

    async def token_generator():
        index = 0
        sent_any = False

        _pending_meta: list[bytes] = []
        _sources: dict[str, str] = {}

        def _emit_process_line(obj):
            nonlocal index
            if isinstance(obj, dict):
                payload = {"process": obj.get("message", ""), **{k: v for k, v in obj.items() if k != "message"}}
            else:
                payload = {"process": str(obj)}
            _pending_meta.append((json.dumps({"meta": payload}) + "\n").encode("utf-8"))

        try:
            print(f"[Runner] Starting token generation with state: {initial_state.get('route', 'unknown_route')}")
            # Install a safe placeholder emitter so early calls won't blow up; replaced later when we know how we'll stream
            set_process_emitter(lambda obj: _emit_process_line(obj))

            # For routed graph, we need to handle streaming differently
            # Try streaming first, if it fails fall back to invoke
            try:
                print(f"[Runner] Attempting streaming with agent type: {type(agent)}")
                if hasattr(agent, "astream_events"):
                    # streaming via astream_events
                    def _emitter(obj):
                        payload = obj if isinstance(obj, dict) else {"event": "process", "message": str(obj)}
                        _pending_meta.append((json.dumps(payload) + "\n").encode("utf-8"))
                        try:
                            if isinstance(payload, dict) and payload.get("event") == "source":
                                url = str(payload.get("url") or "").strip()
                                title = str(payload.get("title") or url).strip()
                                if url:
                                    _sources[url] = title
                        except Exception:
                            pass

                    set_process_emitter(_emitter)
                    # Emit an initial meta with chat_id if present
                    if chat_id:
                        _pending_meta.append((json.dumps({"meta": {"chat_id": chat_id}}) + "\n").encode("utf-8"))

                    async for event in agent.astream_events(initial_state, version="v1"):
                        while _pending_meta:
                            yield _pending_meta.pop(0)
                        try:
                            text = normalize_stream_event(event)
                            if text:
                                sent_any = True
                                yield (json.dumps({"event": "token", "text": text, "index": index}) + "\n").encode("utf-8")
                                index += 1
                        except Exception as stream_err:
                            print(f"[Runner] Stream event processing error: {stream_err}, event type: {type(event)}")
                            raise stream_err

                elif hasattr(agent, "astream"):
                    def _emitter(obj):
                        payload = obj if isinstance(obj, dict) else {"event": "process", "message": str(obj)}
                        _pending_meta.append((json.dumps(payload) + "\n").encode("utf-8"))
                        try:
                            if isinstance(payload, dict) and payload.get("event") == "source":
                                url = str(payload.get("url") or "").strip()
                                title = str(payload.get("title") or url).strip()
                                if url:
                                    _sources[url] = title
                        except Exception:
                            pass

                    set_process_emitter(_emitter)
                    if chat_id:
                        _pending_meta.append((json.dumps({"meta": {"chat_id": chat_id}}) + "\n").encode("utf-8"))

                    async for event in agent.astream(initial_state, stream_mode="values"):
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
                else:
                    # Graph doesn't support streaming, use invoke
                    raise Exception("No streaming support")
            except Exception as e:
                print(f"[Runner] Streaming failed: {e}, falling back to invoke")
                sent_any = False
        except Exception:
            sent_any = False

        # If streaming didn't work, use invoke and stream the result
        if not sent_any:
            try:
                print("[Runner] Using invoke fallback for streaming")
                # Ensure any pending meta is flushed later
                try:
                    while _pending_meta:
                        yield _pending_meta.pop(0)
                except Exception:
                    pass

                # **Important: install an emitter that appends to _pending_meta so chart modules'
                # emit_process(...) calls are captured during invoke.**
                def _invoke_emitter(obj):
                    if isinstance(obj, dict):
                        payload = {"process": obj.get("message", ""), **{k: v for k, v in obj.items() if k != "message"}}
                    else:
                        payload = {"process": str(obj)}
                    _pending_meta.append((json.dumps({"meta": payload}) + "\n").encode("utf-8"))
                    try:
                        if isinstance(obj, dict) and obj.get("event") == "source":
                            url = str(obj.get("url") or "").strip()
                            title = str(obj.get("title") or url).strip()
                            if url:
                                _sources[url] = title
                    except Exception:
                        pass

                set_process_emitter(_invoke_emitter)

                print(f"[Runner] Invoking with state: {len(initial_state['messages'])} messages")
                resp = agent.invoke(initial_state)
                print(f"[Runner] Invoke response type: {type(resp)}")

                # Handle different response types
                if isinstance(resp, dict):
                    messages = resp.get("messages", [])
                    if messages:
                        last_message = messages[-1]
                        raw_content = getattr(last_message, "content", "") if hasattr(last_message, "content") else str(last_message)
                    else:
                        raw_content = ""
                elif isinstance(resp, str):
                    raw_content = resp
                else:
                    raw_content = str(resp)

                if raw_content:
                    print(f"[Runner] Got invoke response, length: {len(raw_content)}")

                    formatted_content = format_financial_content(raw_content)

                    # Flush pending meta before streaming tokens
                    try:
                        while _pending_meta:
                            yield _pending_meta.pop(0)
                    except Exception:
                        pass

                    # Stream the formatted content in chunks
                    chunk_size = 400
                    for i in range(0, len(formatted_content), chunk_size):
                        piece = formatted_content[i:i+chunk_size]
                        if piece:
                            yield (json.dumps({"event": "token", "text": piece, "index": index}) + "\n").encode("utf-8")
                            index += 1

                    # After main content, append citations if any
                    if _sources:
                        citations_header = "\n## Citations\n"
                        yield (json.dumps({"event": "token", "text": citations_header, "index": index}) + "\n").encode("utf-8")
                        index += 1
                        for url, title in _sources.items():
                            line = f"- [{title}]({url})\n"
                            yield (json.dumps({"event": "token", "text": line, "index": index}) + "\n").encode("utf-8")
                            index += 1
                else:
                    print("[Runner] No content received from invoke")
                    yield (json.dumps({"event": "token", "text": "Sorry, I encountered an issue processing your request.", "index": index}) + "\n").encode("utf-8")
                    index += 1
            except Exception as e:
                print(f"[Runner] Invoke fallback failed: {e}")
                yield (json.dumps({"event": "token", "text": "Sorry, I encountered an issue processing your request.", "index": index}) + "\n").encode("utf-8")
                index += 1

        # If streaming branch succeeded and we have sources, append citations before end
        if sent_any and _sources:
            citations_header = "\n## Citations\n"
            yield (json.dumps({"event": "token", "text": citations_header, "index": index}) + "\n").encode("utf-8")
            index += 1
            for url, title in _sources.items():
                line = f"- [{title}]({url})\n"
                yield (json.dumps({"event": "token", "text": line, "index": index}) + "\n").encode("utf-8")
                index += 1

        # Final end marker
        yield (json.dumps({"event": "end"}) + "\n").encode("utf-8")

    try:
        headers = {
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        }
        return StreamingResponse(token_generator(), media_type="text/plain", headers=headers)
    except Exception as streaming_error:
        error_msg = f"Streaming error: {str(streaming_error)}"
        print(f"[Runner] {error_msg}")
        # Return error response
        async def error_generator():
            yield (json.dumps({"event": "token", "text": error_msg, "index": 0}) + "\n").encode("utf-8")
            yield (json.dumps({"event": "end"}) + "\n").encode("utf-8")
        return StreamingResponse(error_generator(), media_type="text/plain")
