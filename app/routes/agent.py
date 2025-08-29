from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

from app.agents.super.runner import run_super_agent

router = APIRouter(prefix="/agent", tags=["agent"])


class AgentPayload(BaseModel):
    question: str
    chat_id: Optional[str] = None
    user_id: Optional[str] = None
    reset_memory: Optional[bool] = False


@router.post("")
def run_agent(payload: AgentPayload):
    try:
        # Use chat_id (preferred) or user_id to scope memory to this conversation
        thread_id = payload.chat_id or payload.user_id
        # Optional memory reset on this call
        if payload.reset_memory and thread_id:
            try:
                from app.tools.memory_store import global_memory_store
                global_memory_store().clear(chat_id=thread_id, user_id=None)
            except Exception:
                pass
        result = run_super_agent(payload.question, memory={}, thread_id=thread_id)
        # Extract compact memory view expected by clients
        mem: Dict[str, Any] = result.get("memory") or {}
        compact_mem = {
            "chat_id": thread_id,
            "last_company": mem.get("last_company") or mem.get("company"),
            "last_ticker": mem.get("last_ticker") or mem.get("ticker"),
            "last_intent": mem.get("last_intent"),
        }
        # Also expose resolved company/ticker in the response for convenience
        return {
            "answer": result.get("output", ""),
            "company": result.get("company"),
            "ticker": result.get("ticker"),
            "exchange": result.get("exchange"),
            "memory": compact_mem,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
