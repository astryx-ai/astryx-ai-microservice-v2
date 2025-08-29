from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.services.agent import agent_answer

router = APIRouter(prefix="/chat", tags=["chat"])

class ChatPayload(BaseModel):
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


@router.post("")
def chat(payload: ChatPayload) -> ChatResponse:
    try:
        answer = agent_answer(payload.query)
        return ChatResponse(success=True, data=ChatResponseData(response=answer))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
