from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict
from typing import Optional, Dict, Any
from app.graph.runner import run_chat as run_chat_graph

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
        result_graph = run_chat_graph({
            "query": payload.query,
            "user_id": payload.user_id,
            "chat_id": payload.chat_id,
        }, memory=mem)
        # run_chat_graph respects feature flags; if disabled it calls legacy under the hood
        result = {"output": result_graph.get("response", "")}
        if key:
            _MEM[key] = mem
        return ChatResponse(success=True, data=ChatResponseData(response=result.get("output") or ""))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Compatibility alias for clients calling POST /chat/message
@router.post("/message")
def chat_message(payload: ChatPayload) -> ChatResponse:
    return chat(payload)
