from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict
from typing import Optional, Dict, Any
from app.services.super_agent import run_super_agent

router = APIRouter(prefix="/chat", tags=["chat"])

class ChatPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")
    query: str
    user_id: Optional[str] = None
    chat_id: Optional[str] = None
    

class ChatResponseData(BaseModel):
    response: str
    chart_data: Optional[dict] = None
    tokens_used: Optional[int] = None
    cost: Optional[float] = None


class ChatResponse(BaseModel):
    success: bool
    data: ChatResponseData


_MEM: Dict[str, Dict[str, Any]] = {}


@router.post("")
def chat(payload: ChatPayload) -> ChatResponse:
    try:
        key = payload.chat_id or payload.user_id or ""
        mem = _MEM.get(key, {}) if key else {}
        result = run_super_agent(payload.query, memory=mem)
        if key:
            _MEM[key] = mem
        return ChatResponse(success=True, data=ChatResponseData(response=result.get("output") or ""))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
