from fastapi import APIRouter, HTTPException, Response
import logging
import os
from pydantic import BaseModel, ConfigDict
from typing import Optional, Any
from app.graph.runner import run_chat as run_chat_graph
from app.agents.super.runner import run_super_agent

try:
    # Optional memory store for clear operation; keep import lazy-friendly
    from app.tools.memory_store import global_memory_store
except Exception:  # pragma: no cover
    global_memory_store = None  # type: ignore

router = APIRouter(prefix="/chat", tags=["chat"])
# Deprecated shim routers colocated for compactness
agent_router = APIRouter(prefix="/agent", tags=["agent"])  # deprecated shim
super_router = APIRouter(prefix="/super", tags=["super-agent"])  # deprecated shim

class ChatPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")
    # Unify payloads: accept both query and question
    query: Optional[str] = None
    question: Optional[str] = None
    user_id: Optional[str] = None
    chat_id: Optional[str] = None
    # Optional controls (kept for programmatic clients; not required for simple query-only usage)
    reset_memory: Optional[bool] = False
    memory_clear: Optional[bool] = False
    

class ChatResponseData(BaseModel):
    response: str
    chart_data: Optional[dict] = None
    charts: Optional[list] = None
    tokens_used: Optional[int] = None
    cost: Optional[float] = None


class ChatResponse(BaseModel):
    success: bool
    data: ChatResponseData



def _is_memory_clear_query(q: str) -> bool:
    qn = (q or "").strip().lower()
    return qn in {"clear memory", "reset memory", "reset chat", "clear conversation", "forget"}


def _is_chart_intent(q: str) -> bool:
    qn = (q or "").lower()
    for k in ("chart", "candle", "candlestick", "ohlc", "line chart", "area chart", "bar chart", "price chart"):
        if k in qn:
            return True
    return False


def _is_finance_intent(q: str) -> bool:
    qn = (q or "").lower()
    finance_words = [
        "stock", "share", "ticker", "nse", "bse", "price", "market cap", "pe", "fundamental", "intraday",
        "earnings", "revenue", "profit", "dividend", "company", "quarter", "results",
    ]
    return any(w in qn for w in finance_words)


@router.post("")
def chat(payload: ChatPayload) -> Any:
    try:
        # Normalize inputs across legacy/merged routes
        text = (payload.query or payload.question or "").strip()
        thread_id = payload.chat_id or payload.user_id

        # Memory clear operation: parameter-based OR natural-language trigger
        if payload.memory_clear or _is_memory_clear_query(text):
            # Previously POST /super/memory/clear; now merged here
            if global_memory_store:
                try:
                    global_memory_store().clear(chat_id=payload.chat_id, user_id=payload.user_id)
                except Exception:
                    pass
            return ChatResponse(success=True, data=ChatResponseData(response="memory cleared"))

        # Optional reset of memory for this conversation
        if payload.reset_memory and thread_id:
            try:
                if global_memory_store:
                    global_memory_store().clear(chat_id=thread_id, user_id=None)
            except Exception:
                pass

        # Ensure chat pipeline is enabled to leverage LangGraph memory/checkpointer
        if os.getenv("GRAPH_CHAT_ENABLED") is None and os.getenv("GRAPH_ENABLED") is None:
            os.environ["GRAPH_CHAT_ENABLED"] = "true"

        # Heuristic dispatch without explicit modes
        # 1) Chart-like queries → graph chat (it yields charts and responses)
        if _is_chart_intent(text):
            result_graph = run_chat_graph({
                "query": text,
                "user_id": payload.user_id,
                "chat_id": payload.chat_id,
            })
            resp = result_graph.get("response", "")
            return ChatResponse(
                success=True,
                data=ChatResponseData(
                    response=resp or "",
                    chart_data=result_graph.get("chart_data"),
                    charts=result_graph.get("charts"),
                ),
            )

        # 2) Finance-like queries → super agent for better company/ticker reasoning
        if _is_finance_intent(text):
            result = run_super_agent(text, memory={}, thread_id=thread_id)
            return ChatResponse(success=True, data=ChatResponseData(response=result.get("output", "")))

        # 3) Default: graph chat
        result_graph = run_chat_graph({
            "query": text,
            "user_id": payload.user_id,
            "chat_id": payload.chat_id,
        })
        resp = result_graph.get("response", "")
        return ChatResponse(
            success=True,
            data=ChatResponseData(
                response=resp or "",
                chart_data=result_graph.get("chart_data"),
                charts=result_graph.get("charts"),
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# Deprecated shim: /agent
# Prefer POST /chat with just { query } (heuristics handle finance intents)
# Kept for compatibility and logs a warning.
class AgentPayload(BaseModel):
    question: str
    chat_id: Optional[str] = None
    user_id: Optional[str] = None
    reset_memory: Optional[bool] = False


@agent_router.post("")
def run_agent_shim(payload: AgentPayload):
    try:
        logging.warning("[DEPRECATED] POST /agent is deprecated. Use POST /chat with { query }.")
        # Forward to unified /chat. Extra fields like mode are ignored by ChatPayload
        payload_unified = ChatPayload(
            question=payload.question,
            chat_id=payload.chat_id,
            user_id=payload.user_id,
            reset_memory=payload.reset_memory,
        )
        res = chat(payload_unified)
        # Preserve previous simplified shape
        try:
            data = getattr(res, "data", None) or (res.get("data") if isinstance(res, dict) else None)
            msg = (getattr(data, "response", None) if data is not None else None) or ((data or {}).get("response") if isinstance(data, dict) else None)
            return {
                "answer": msg or "",
                "company": None,
                "ticker": None,
                "exchange": None,
                "memory": {"chat_id": payload.chat_id or payload.user_id},
            }
        except Exception:
            return {
                "answer": "",
                "company": None,
                "ticker": None,
                "exchange": None,
                "memory": {"chat_id": payload.chat_id or payload.user_id},
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# Deprecated shim: /super/*
class SuperChatPayload(BaseModel):
    query: str
    chat_id: Optional[str] = None
    user_id: Optional[str] = None
    format: Optional[str] = "json"  # "json" | "md"
    reset_memory: Optional[bool] = False


@super_router.post("/chat")
def super_chat_shim(payload: SuperChatPayload):
    try:
        logging.warning("[DEPRECATED] POST /super/chat is deprecated. Use POST /chat with just { query }.")
        # Preserve Markdown behavior for legacy clients
        key = payload.chat_id or payload.user_id
        if payload.reset_memory and key and global_memory_store:
            try:
                global_memory_store().clear(chat_id=key, user_id=None)
            except Exception:
                pass
        result = run_super_agent(payload.query, memory={}, thread_id=key)
        if (payload.format or "json").lower() == "md":
            md = result.get("output") or ""
            return Response(content=md, media_type="text/markdown")
        # JSON shape
        return {"response": result.get("output")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class MemoryClearPayload(BaseModel):
    chat_id: Optional[str] = None
    user_id: Optional[str] = None


@super_router.post("/memory/clear")
def memory_clear_shim(payload: MemoryClearPayload):
    try:
        logging.warning("[DEPRECATED] POST /super/memory/clear is deprecated. Use POST /chat with { memory_clear: true }.")
        if not (payload.chat_id or payload.user_id):
            raise HTTPException(status_code=400, detail="chat_id or user_id required")
        # Delegate to /chat memory clear
        _ = chat(ChatPayload(chat_id=payload.chat_id, user_id=payload.user_id, memory_clear=True))
        return {"ok": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Deprecated compatibility alias: POST /chat/message
# - Sunset window: 30 days from 2025-08-29
# - Logs a warning on every call and forwards to the canonical POST /chat handler
# - Keep response shape identical to the canonical route
@router.post("/message")
def chat_message(payload: ChatPayload) -> ChatResponse:
    logging.warning("[DEPRECATED] POST /chat/message is deprecated and will be removed after the 30-day sunset. Use POST /chat instead.")
    # Forward to canonical handler to avoid duplication and ensure consistent behavior
    return chat(payload)
