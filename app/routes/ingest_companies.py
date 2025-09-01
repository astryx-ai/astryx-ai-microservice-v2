from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from app.services.ingest_companies import ingest_companies as ingest_companies_service


router = APIRouter(prefix="/ingest/companies", tags=["ingest"])


@router.post("")
def ingest_companies(
    limit: Optional[int] = Query(None, description="Limit number of rows ingested for testing"),
    mode: str = Query("upsert", description="upsert | replace | truncate")
):
    try:
        return ingest_companies_service(limit=limit, mode=mode)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
