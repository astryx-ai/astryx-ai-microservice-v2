from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from langchain.schema import Document

from app.services.insight_service import get_company_insights
from app.scrapper.sanitize import clean_text
from app.services.rag import chunk_text, upsert_news

router = APIRouter(prefix="/insight", tags=["insight"])


class InsightItem(BaseModel):
    summary: str
    source: str


class InsightResponse(BaseModel):
    company: str
    insights: List[InsightItem]
    all_sources: List[str]


@router.post("", response_model=InsightResponse)
async def generate_insight(
    company: str = Query(..., description="Company name"),
    limit: int = Query(10, description="Number of news articles to summarize")
):
    try:
        result = await get_company_insights(company, limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    docs: List[Document] = []
    for item in result["insights"]:
        blob = clean_text(item["summary"])
        if not blob:
            continue
        meta: Dict[str, Any] = {
            "company": result["company"],
            "source": item["source"],
            "type": "insight"
        }
        docs.extend(chunk_text(blob, meta))

    # ✅ store in vector DB, but do not return its raw response
    if docs:
        upsert_news(docs)

    # ✅ always return InsightResponse shape
    return InsightResponse(
        company=result["company"],
        insights=result["insights"],
        all_sources=result["all_sources"]
    )
