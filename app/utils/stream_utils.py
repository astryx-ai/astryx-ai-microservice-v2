from contextvars import ContextVar
from typing import Callable, Optional, Dict, Any


_process_emitter: ContextVar[Optional[Callable[[Dict[str, Any]], None]]] = ContextVar("process_emitter", default=None)


def set_process_emitter(emitter: Optional[Callable[[Dict[str, Any]], None]]):
    _process_emitter.set(emitter)


def emit_process(event: Dict[str, Any] | str):
    emitter = _process_emitter.get()
    if not emitter:
        return
    # Normalize to the streaming schema
    if isinstance(event, str):
        payload: Dict[str, Any] = {"event": "process", "message": event}
    elif isinstance(event, dict) and event.get("event") == "chart_data":
        # Handle chart_data events specially
        payload = {"event": "chart_data", "chart": event.get("chart", {})}
    else:
        # Regular process message
        message_text = event.get("message") if isinstance(event, dict) else None
        payload = {"event": "process", "message": str(message_text or "")}
    try:
        emitter(payload)
    except Exception:
        pass


def normalize_stream_event(event) -> str:
    """Extract streamed text token/content from heterogeneous event payloads.
    Returns empty string when no user-visible text is present.
    """
    ev_name = ""
    data_obj = None
    if isinstance(event, dict):
        ev_name = str(event.get("event") or event.get("type") or "")
        data_obj = event.get("data")
    else:
        ev_name = str(getattr(event, "event", "") or getattr(event, "type", ""))
        data_obj = getattr(event, "data", None) or event

    if ev_name not in ("on_chat_model_stream", "on_llm_stream"):
        return ""

    if isinstance(data_obj, dict):
        possible_chunk = data_obj.get("chunk") or data_obj.get("token") or ""
        if hasattr(possible_chunk, "content"):
            return getattr(possible_chunk, "content", None) or ""
        if hasattr(possible_chunk, "delta"):
            return getattr(possible_chunk, "delta", None) or ""
        if isinstance(possible_chunk, str):
            return possible_chunk
        return ""

    return getattr(data_obj, "content", None) or getattr(data_obj, "delta", None) or ""


