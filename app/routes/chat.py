from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Literal, Dict
from app.services.rag import rag_answer

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatPayload(BaseModel):
    question: str
    domain: Literal["news", "stocks", "both"] = "news"
    ticker: Optional[str] = None
    company: Optional[str] = None
    

@router.post("")
def chat(payload: ChatPayload):
    try:
        filters: Dict[str, str] = {}
        if payload.ticker:
            filters["ticker"] = payload.ticker.upper()
        if payload.company:
            filters["company"] = payload.company
        answer = rag_answer(
            payload.question, domain=payload.domain, filters=filters or None)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
