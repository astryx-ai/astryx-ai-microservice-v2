from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.graph.runner import run_news

router = APIRouter(prefix="/ingest/news", tags=["ingest"])


class IngestNewsPayload(BaseModel):
    ticker: str
    company: Optional[str] = None
    limit: int = 20


@router.post("")
def ingest_news(payload: IngestNewsPayload):
    """Ingest news using legacy flow or LangGraph based on feature flags.

    Response stays the same: {"ingested": int}
    """
    try:
        result = run_news({
            "ticker": payload.ticker,
            "company": payload.company,
            "limit": payload.limit,
        })
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
