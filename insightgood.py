from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import List
from app.services.insight_service import get_company_insights

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
    # Await the async service function
    result = await get_company_insights(company, limit)
    return result
