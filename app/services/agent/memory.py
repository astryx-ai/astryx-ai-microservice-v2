from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage

from app.services.db.messages import fetch_recent_messages, fetch_relevant_messages
from app.services.llms.azure_openai import embedder
from app.stream_utils import emit_process


def _deduplicate_messages(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for it in items:
        mid = it.get("id")
        if mid and mid not in seen:
            seen.add(mid)
            out.append(it)
    return out


def build_langchain_messages(raw_items: List[Dict[str, Any]]) -> List[Any]:
    # Reverse to chronological order (oldest first) for conversation flow
    items = list(reversed(raw_items))
    lc_messages: List[Any] = []
    for it in items:
        content = it.get("content") or ""
        is_ai = bool(it.get("is_ai"))
        if is_ai:
            lc_messages.append(AIMessage(content=content))
        else:
            lc_messages.append(HumanMessage(content=content))
    return lc_messages


def get_context(chat_id: str, query_text: str, recency_limit: int = 10, retrieval_limit: int = 5) -> List[Any]:
    emit_process({"message": "Looking through memory"})
    recent = fetch_recent_messages(chat_id, limit=recency_limit)
    # Create embedding for semantic retrieval
    embedding_model = embedder()
    emit_process({"message": "Embedding query for semantic retrieval"})
    query_embedding = embedding_model.embed_query(query_text)
    emit_process({"message": "Searching memory for relevant context"})
    relevant = fetch_relevant_messages(chat_id, query_embedding=query_embedding, limit=retrieval_limit)
    merged = _deduplicate_messages(recent + relevant)
    return build_langchain_messages(merged)


