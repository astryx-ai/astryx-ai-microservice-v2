from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.agent import agent_answer


router = APIRouter(prefix="/agent", tags=["agent"])


class AgentPayload(BaseModel):
    question: str


@router.post("")
def run_agent(payload: AgentPayload):
    try:
        answer = agent_answer(payload.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
