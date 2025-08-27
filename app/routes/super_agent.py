from fastapi import APIRouter, HTTPException, Response
from typing import Optional, Dict, Any
from pydantic import BaseModel

from app.services.super_agent import run_super_agent

router = APIRouter(prefix="/super", tags=["super-agent"])

_MEM: Dict[str, Dict[str, Any]] = {}


class SuperChatPayload(BaseModel):
    query: str
    chat_id: Optional[str] = None
    user_id: Optional[str] = None
    format: Optional[str] = "json"  # "json" | "md"


@router.post("/chat")
def super_chat(payload: SuperChatPayload):
    """Run the Yahoo-based SuperAgent. Optional memory via chat_id or user_id."""
    try:
        key = payload.chat_id or payload.user_id  # prefer stable key if available
        mem: Dict[str, Any] = _MEM.get(key, {}) if key else {}
        result = run_super_agent(payload.query, memory=mem)
        if key:
            _MEM[key] = mem  # persist only when a stable key exists
        if (payload.format or "json").lower() == "md":
            md = result.get("output") or ""
            return Response(content=md, media_type="text/markdown")
        return {"response": result.get("output")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
