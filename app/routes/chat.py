from fastapi import APIRouter, HTTPException
import os
from pydantic import BaseModel, ConfigDict
from typing import Optional
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
    charts: Optional[list] = None
    tokens_used: Optional[int] = None
    cost: Optional[float] = None


class ChatResponse(BaseModel):
    success: bool
    data: ChatResponseData



@router.post("")
def chat(payload: ChatPayload) -> ChatResponse:
    try:
        # Ensure chat pipeline is enabled to leverage LangGraph memory/checkpointer
        if os.getenv("GRAPH_CHAT_ENABLED") is None and os.getenv("GRAPH_ENABLED") is None:
            os.environ["GRAPH_CHAT_ENABLED"] = "true"
        result_graph = run_chat_graph({
            "query": payload.query,
            "user_id": payload.user_id,
            "chat_id": payload.chat_id,
        })
        # run_chat_graph respects feature flags; if disabled it calls legacy under the hood
        resp = result_graph.get("response", "")
        chart_payload = result_graph.get("chart_data")
        charts_payload = result_graph.get("charts")
        return ChatResponse(success=True, data=ChatResponseData(response=resp or "", chart_data=chart_payload, charts=charts_payload))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Compatibility alias for clients calling POST /chat/message
@router.post("/message")
def chat_message(payload: ChatPayload) -> ChatResponse:
    return chat(payload)
