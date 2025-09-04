from langchain_core.messages import HumanMessage, SystemMessage
from starlette.responses import StreamingResponse
import json

from .builder import build_agent
from .memory import get_context
from app.stream_utils import normalize_stream_event, set_process_emitter
from app.services.agent_tools.formatter import format_financial_content

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
    
    # Get context and build state
    context_msgs = get_context(chat_id, question) if chat_id else []
    state = {"messages": context_msgs + [system_msg, user_msg]}
    
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
    
    # Apply formatting
    formatted_content = format_financial_content(raw_content)
    return formatted_content


# Agent responses with streaming
async def agent_stream_response(question: str, user_id: str | None = None, chat_id: str | None = None):
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
    
    print(f"[Runner] Initial state setup - context_msgs: {len(context_msgs)}, system: {type(system)}, user: {type(user)}")

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
            # Install process emitter into context so memory/tools can publish
            set_process_emitter(lambda obj: (_ for _ in ()).throw(StopIteration))  # placeholder in case called before loop
            
            # For routed graph, we need to handle streaming differently
            # Try streaming first, if it fails fall back to invoke
            try:
                print(f"[Runner] Attempting streaming with agent type: {type(agent)}")
                if hasattr(agent, "astream_events"):
                    # Rebind to emit via outer generator "yield" using event schema
                    def _emitter(obj):
                        nonlocal index
                        payload = obj if isinstance(obj, dict) else {"event": "process", "message": str(obj)}
                        raise_yield = (json.dumps(payload) + "\n").encode("utf-8")
                        nonlocal _pending_meta
                        _pending_meta.append(raise_yield)

                    set_process_emitter(_emitter)
                    # Emit an initial meta with chat_id if present
                    if chat_id:
                        _pending_meta.append((json.dumps({"meta": {"chat_id": chat_id}}) + "\n").encode("utf-8"))
                    
                    async for event in agent.astream_events({"messages": context_msgs + [system, user]}, version="v1"):
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
                        nonlocal index
                        payload = obj if isinstance(obj, dict) else {"event": "process", "message": str(obj)}
                        _pending_meta.append((json.dumps(payload) + "\n").encode("utf-8"))

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
                # Flush any pending meta first
                try:
                    while _pending_meta:
                        yield _pending_meta.pop(0)
                except Exception:
                    pass
                    
                state_to_invoke = {"messages": context_msgs + [system, user]}
                print(f"[Runner] Invoking with state: {len(state_to_invoke['messages'])} messages")
                resp = agent.invoke(state_to_invoke)
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
                    # Apply intelligent formatting
                    formatted_content = format_financial_content(raw_content)
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

    headers = {
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(token_generator(), media_type="text/plain", headers=headers)
