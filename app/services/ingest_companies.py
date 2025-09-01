from typing import List, Dict, Any, Optional, Set

from supabase import Client

from app.scrapper.companies_scraper import get_companies
from app.config import settings
from app.services.db.supabase import get_supabase_client, get_psycopg_connection


def _supabase() -> Client:
    return get_supabase_client()


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
        with get_psycopg_connection() as conn:
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


def ingest_companies(limit: Optional[int], mode: str) -> Dict[str, Any]:
    companies = get_companies()
    if limit is not None:
        companies = companies[: max(0, int(limit))]
    if not companies:
        return {"ingested": 0}

    supa = _supabase()

    # Prepare payload
    payload: List[Dict[str, Any]] = [_coerce_record(x) for x in companies if x.get("isin")]
    if not payload:
        return {"ingested": 0}

    CHUNK = 1000

    if mode == "truncate":
        _truncate_companies_table()
        total = 0
        for i in range(0, len(payload), CHUNK):
            batch = payload[i : i + CHUNK]
            supa.table("companies").insert(batch).execute()
            total += len(batch)
        return {"ingested": total, "mode": mode}

    if mode == "replace":
        # Upsert first to ensure all present rows exist, then delete rows no longer present
        total = 0
        for i in range(0, len(payload), CHUNK):
            batch = payload[i : i + CHUNK]
            supa.table("companies").upsert(batch, on_conflict="isin").execute()
            total += len(batch)
        deleted = _delete_missing_isins(supa, {p["isin"] for p in payload})
        return {"ingested": total, "deleted_missing": deleted, "mode": mode}

    # Default: idempotent upsert-only
    total = 0
    for i in range(0, len(payload), CHUNK):
        batch = payload[i : i + CHUNK]
        supa.table("companies").upsert(batch, on_conflict="isin").execute()
        total += len(batch)
    return {"ingested": total, "mode": mode}


