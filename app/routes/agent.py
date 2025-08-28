from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.agents.super.runner import run_super_agent

router = APIRouter(prefix="/agent", tags=["agent"])


class AgentPayload(BaseModel):
    question: str


@router.post("")
def run_agent(payload: AgentPayload):
    try:
        result = run_super_agent(payload.question, memory={})
        return {"answer": result.get("output", "")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
