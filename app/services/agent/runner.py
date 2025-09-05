from langchain_core.messages import HumanMessage, SystemMessage
from starlette.responses import StreamingResponse
import json

from app.services.agent_tools.helper_tools import decide_route
from .builder import build_agent
from .memory import get_context
from app.utils.stream_utils import normalize_stream_event, set_process_emitter, emit_process
from app.services.agent_tools.formatter import format_financial_content


def _extract_chart_data(content: str):
    """
    Robust extractor: finds the first JSON object whose top-level "type" == "bar-standard",
    using a brace walker that respects JSON strings and escapes. Returns (parsed_json, remaining_text).
    If none found, returns (None, original_content).
    Also removes a one-line header immediately preceding the JSON if it looks like a chart header.
    """
    try:
        text = content or ""
        # helper to remove a single-line header preceding `start_idx`
        def remove_preceding_header(s: str, start_idx: int):
            prev_nl = s.rfind('\n', 0, start_idx)
            if prev_nl == -1:
                prev_nl = 0
            header = s[prev_nl:start_idx].strip().lower()
            keywords = [
                'below is', 'json chart', 'chart visualization', 'below is the json',
                'below is the chart', 'chart data', 'json chart visualization',
                'below is the json chart visualization'
            ]
            if (any(kw in header for kw in keywords) and len(header) < 160) or (header.endswith(':') and len(header) < 80):
                return prev_nl
            return start_idx

        i = 0
        n = len(text)
        while i < n:
            # find next opening brace
            start = text.find('{', i)
            if start == -1:
                break

            # walk forward to find matching closing brace, respecting strings/escapes
            j = start
            brace_count = 0
            in_string = False
            escape = False
            end_index = -1
            while j < n:
                ch = text[j]
                if ch == '"' and not escape:
                    in_string = not in_string
                    j += 1
                    escape = False
                    continue
                if in_string:
                    if ch == '\\' and not escape:
                        escape = True
                    else:
                        escape = False
                    j += 1
                    continue
                # not in string
                if ch == '{':
                    brace_count += 1
                elif ch == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_index = j
                        break
                j += 1

            if end_index == -1:
                # couldn't find a matching end for this opening brace; move past it
                i = start + 1
                continue

            candidate = text[start:end_index + 1].strip()
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict) and parsed.get("type") == "bar-standard":
                    # remove any short header immediately before the JSON
                    new_start = remove_preceding_header(text, start)
                    remaining = (text[:new_start] + text[end_index + 1:]).strip()
                    # collapse excessive blank lines
                    remaining = '\n\n'.join([p.strip() for p in remaining.splitlines()]).strip()
                    return parsed, remaining
            except Exception:
                # not valid JSON here -- continue scanning
                pass

            i = start + 1

    except Exception as e:
        print(f"[Runner] Chart extraction error: {e}")

    return None, content


def _process_chart_content(content: str, route: str):
    """Process content and extract chart data if it's from chart_viz route."""
    if route == "chart_viz":
        chart_data, remaining_content = _extract_chart_data(content)
        return chart_data, remaining_content
    return None, content


# Agent responses without streaming
def agent_answer(question: str, user_id: str | None = None, chat_id: str | None = None) -> str:
    """Run the routed LangGraph agent and return the final answer."""
    print(f"[Runner] agent_answer invoked | user_id={user_id}, chat_id={chat_id}")

    # Build the routed graph
    graph = build_agent()

    # Simple system and user messages for the graph
    system_msg = SystemMessage(content=(
        "You are a financial AI assistant. Use the available search tools to find relevant information "
        "and present it with clear structure, tables for numerical data, and proper citations."
    ))
    user_msg = HumanMessage(content=f"Task: {question}")

    # Get context and build state with pre-computed routing
    context_msgs = get_context(chat_id, question) if chat_id else []

    user_and_ai_messages = [msg for msg in context_msgs + [user_msg] if hasattr(msg, 'type') and msg.type in ["human", "ai"]]
    has_context = len(user_and_ai_messages) > 2

    print("[Runner] Pre-computing routing decision for non-streaming")
    try:
        available_routes = ["standard", "deep_research", "chart_viz"]
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
    }
    print(f"[Runner] Pre-computed route: {route}")

    print("[Runner] agent_answer executing routed graph")
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

    # If this route was chart_viz, extract chart JSON and emit it via emit_process.
    if state.get("route") == "chart_viz":
        chart, remaining = _process_chart_content(raw_content, "chart_viz")
        if chart:
            try:
                # emit chart_data meta for non-streaming clients that listen to process events
                emit_process({"event": "chart_data", "chart": chart})
            except Exception as e:
                print(f"[Runner] emit_process failed in agent_answer: {e}")
        formatted_content = format_financial_content(remaining)
    else:
        formatted_content = format_financial_content(raw_content)

    return formatted_content


# Agent responses with streaming
async def agent_stream_response(question: str, user_id: str | None = None, chat_id: str | None = None):
    try:
        print(f"[Runner] agent_stream_response invoked | user_id={user_id}, chat_id={chat_id}")

        # Build the routed graph
        agent = build_agent()

        # Simple system and user messages for the graph
        system = SystemMessage(content=(
            "You are a financial AI assistant. Use the available search tools to find relevant information "
            "and present it with clear structure, tables for numerical data, and proper citations."
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
        available_routes = ["standard", "deep_research", "chart_viz"]
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
        "context": user_and_ai_messages
    }
    print(f"[Runner] Pre-computed route: {chosen_route}")

    async def token_generator():
        index = 0
        sent_any = False

        _pending_meta: list[bytes] = []

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

                    # If route is chart_viz, attempt to extract chart JSON locally in case the module didn't
                    # or its emit_process wasn't captured for some reason.
                    if initial_state.get("route") == "chart_viz":
                        chart, remaining = _process_chart_content(raw_content, "chart_viz")
                        if chart:
                            try:
                                # Emit as meta so client can receive chart data as an event before tokens
                                _pending_meta.append((json.dumps({"meta": {"event": "chart_data", "chart": chart}}) + "\n").encode("utf-8"))
                                print("[Runner] Injected chart_data into pending meta")
                            except Exception as e:
                                print(f"[Runner] Failed to inject chart into meta: {e}")
                        formatted_content = format_financial_content(remaining)
                    else:
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
                else:
                    print("[Runner] No content received from invoke")
                    yield (json.dumps({"event": "token", "text": "Sorry, I encountered an issue processing your request.", "index": index}) + "\n").encode("utf-8")
                    index += 1
            except Exception as e:
                print(f"[Runner] Invoke fallback failed: {e}")
                yield (json.dumps({"event": "token", "text": "Sorry, I encountered an issue processing your request.", "index": index}) + "\n").encode("utf-8")
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
