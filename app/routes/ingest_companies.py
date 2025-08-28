from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any, Optional, Set
from supabase import Client

from app.services.scrapers.companies_scraper import get_companies
from app.tools.config import settings
from supabase import create_client
import psycopg
from app.graph.runner import run_companies


router = APIRouter(prefix="/ingest/companies", tags=["ingest"])


def _supabase() -> Client:
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)


def _coerce_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "company_name": rec.get("company_name"),
        "nse_symbol": rec.get("nse_symbol"),
        "bse_code": rec.get("bse_code"),
        "bse_symbol": rec.get("bse_symbol"),
        "isin": rec.get("isin"),
        "industry": rec.get("industry"),
        "status": rec.get("status"),
        "market_cap": rec.get("market_cap"),
    }


def _truncate_companies_table() -> None:
    # Prefer direct DB truncate for cascade when DATABASE_URL is available
    dsn = getattr(settings, "DATABASE_URL", None)
    if dsn:
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE companies CASCADE;")
                conn.commit()
        return
    # Fallback: delete all rows via Supabase (row-by-row deletion still honors FK ON DELETE CASCADE)
    supa = _supabase()
    supa.table("companies").delete().neq("id", "").execute()


def _fetch_all_isins(supa: Client) -> Set[str]:
    keep: Set[str] = set()
    start = 0
    step = 1000
    while True:
        resp = supa.table("companies").select("isin").range(start, start + step - 1).execute()
        rows = resp.data or []
        if not rows:
            break
        for r in rows:
            v = (r or {}).get("isin")
            if v:
                keep.add(v)
        if len(rows) < step:
            break
        start += step
    return keep


def _delete_missing_isins(supa: Client, keep_isins: Set[str]) -> int:
    existing = _fetch_all_isins(supa)
    to_delete = [v for v in existing if v not in keep_isins]
    deleted = 0
    CHUNK = 1000
    for i in range(0, len(to_delete), CHUNK):
        batch = to_delete[i : i + CHUNK]
        if not batch:
            continue
        supa.table("companies").delete().in_("isin", batch).execute()
        deleted += len(batch)
    # Also remove rows with NULL ISIN (legacy)
    supa.table("companies").delete().is_("isin", None).execute()
    return deleted


@router.post("")
def ingest_companies(
    limit: Optional[int] = Query(None, description="Limit number of rows ingested for testing"),
    mode: str = Query("upsert", description="upsert | replace | truncate")
):
    try:
        return run_companies({"limit": limit, "mode": mode})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


