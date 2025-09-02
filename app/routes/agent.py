from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, AliasChoices

from app.services.agent.runner import agent_answer, agent_stream_response


router = APIRouter(prefix="/agent", tags=["agent"])


class AgentPayload(BaseModel):
    question: str = Field(validation_alias=AliasChoices("question", "query"))
    user_id: str | None = None
    chat_id: str | None = None


@router.post("")
def run_agent(payload: AgentPayload):
    try:
        answer = agent_answer(payload.question, user_id=payload.user_id, chat_id=payload.chat_id)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class AgentStreamPayload(BaseModel):
    question: str = Field(validation_alias=AliasChoices("question", "query"))
    user_id: str | None = None
    chat_id: str | None = None


@router.post("/stream")
async def run_agent_stream(payload: AgentStreamPayload):
    try:
        return await agent_stream_response(payload.question, user_id=payload.user_id, chat_id=payload.chat_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
