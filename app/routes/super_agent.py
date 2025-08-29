from fastapi import APIRouter, HTTPException, Response
from typing import Optional
from pydantic import BaseModel

from app.agents.super.runner import run_super_agent

router = APIRouter(prefix="/super", tags=["super-agent"])


class SuperChatPayload(BaseModel):
    query: str
    chat_id: Optional[str] = None
    user_id: Optional[str] = None
    format: Optional[str] = "json"  # "json" | "md"
    reset_memory: Optional[bool] = False


@router.post("/chat")
def super_chat(payload: SuperChatPayload):
    """Run the Yahoo-based SuperAgent. Optional memory via chat_id or user_id."""
    try:
        key = payload.chat_id or payload.user_id  # prefer stable key if available
        if payload.reset_memory and key:
            try:
                from app.tools.memory_store import global_memory_store
                global_memory_store().clear(chat_id=key, user_id=None)
            except Exception:
                pass
        result = run_super_agent(payload.query, memory={}, thread_id=key)
        if (payload.format or "json").lower() == "md":
            md = result.get("output") or ""
            return Response(content=md, media_type="text/markdown")
        return {"response": result.get("output")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Lightweight memory management endpoints
class MemoryClearPayload(BaseModel):
    chat_id: Optional[str] = None
    user_id: Optional[str] = None


@router.post("/memory/clear")
def memory_clear(payload: MemoryClearPayload):
    try:
        key = payload.chat_id or payload.user_id
        if not key:
            raise HTTPException(status_code=400, detail="chat_id or user_id required")
        from app.tools.memory_store import global_memory_store
        global_memory_store().clear(chat_id=payload.chat_id, user_id=payload.user_id)
        return {"ok": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
