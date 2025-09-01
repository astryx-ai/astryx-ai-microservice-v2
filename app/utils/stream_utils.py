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


