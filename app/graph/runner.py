from __future__ import annotations
import os
from typing import Any, Dict
import json
import asyncio

from app.graph.state import BaseState, mk_state
from app.graph.checkpoints import default_saver

from app.graph.pipelines import news as news_pipeline
from app.graph.pipelines import chart as chart_pipeline
from app.graph.pipelines import chat as chat_pipeline
from app.graph.pipelines import companies as companies_pipeline


def is_graph_enabled(scope: str) -> bool:
    # Allow per-pipeline flags: GRAPH_NEWS_ENABLED etc., or global GRAPH_ENABLED
    global_flag = os.getenv("GRAPH_ENABLED", "false").lower() in ("1", "true", "yes", "on")
    scoped_flag = os.getenv(f"GRAPH_{scope.upper()}_ENABLED", "").lower() in ("1", "true", "yes", "on")
    return global_flag or scoped_flag


def run_news(inputs: Dict[str, Any]) -> Dict[str, Any]:
    state: BaseState = mk_state("news", inputs, mem_key=None)
    if not is_graph_enabled("news"):
        # Legacy path: inline replicate of route logic using existing services
        # Import locally to avoid import cycles
        from app.services.scrapers.scrape_news import get_news
        from app.services.scrapers.sanitize import clean_text
        from app.tools.rag import chunk_text, upsert_news
        items = get_news(inputs.get("ticker") or inputs.get("company") or inputs.get("query") or "") or []
        items = items[: int(inputs.get("limit") or 20)]
        docs = []
        for it in items:
            title = it.get("title", "")
            text = it.get("text") or it.get("summary") or ""
            blob = clean_text(f"{title}. {text}")
            if not blob:
                continue
            meta = {
                "ticker": (inputs.get("ticker") or inputs.get("symbol") or "").upper(),
                "company": inputs.get("company") or "",
                "source": it.get("url") or it.get("source") or "news",
                "title": title,
                "type": "news",
            }
            docs.extend(chunk_text(blob, meta))
        if docs:
            upsert_news(docs)
        return {"ingested": len(docs)}

    saver = default_saver()
    final = news_pipeline.run(state, saver=saver)
    docs_count = ((final.get("outputs") or {}).get("counts") or {}).get("docs", 0)
    return {"ingested": int(docs_count or 0)}


def run_chart(inputs: Dict[str, Any]) -> Dict[str, Any]:
    state: BaseState = mk_state("chart", inputs, mem_key=None)
    if not is_graph_enabled("chart"):
        # Legacy path: call builder directly
        from app.features.charts.builder import build_chart
        try:
            payload = asyncio.run(build_chart(
                query=inputs.get("query", ""),
                symbol=inputs.get("symbol", ""),
                chart_type=inputs.get("chart_type", ""),
                range_=inputs.get("range") or inputs.get("range_", "1d"),
                interval=inputs.get("interval", "5m"),
                title=inputs.get("title", ""),
                description=inputs.get("description", ""),
            ))
        except RuntimeError:
            payload = asyncio.get_event_loop().run_until_complete(build_chart(
                query=inputs.get("query", ""),
                symbol=inputs.get("symbol", ""),
                chart_type=inputs.get("chart_type", ""),
                range_=inputs.get("range") or inputs.get("range_", "1d"),
                interval=inputs.get("interval", "5m"),
                title=inputs.get("title", ""),
                description=inputs.get("description", ""),
            ))  # type: ignore
        json_str = payload if isinstance(payload, str) else json.dumps(payload or {})
        return {"json": json_str, "content_type": "application/json"}
    saver = default_saver()
    final = chart_pipeline.run(state, saver=saver)
    payload = (final.get("outputs") or {}).get("chart") or {}
    json_str = payload if isinstance(payload, str) else json.dumps(payload or {})
    return {"json": json_str, "content_type": "application/json"}


async def run_chart_async(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Async variant for gRPC servers; uses legacy builder directly to avoid nested loops.

    We still respect the same output shape and flags are ignored here for safety.
    """
    from app.features.charts.builder import build_chart
    payload = await build_chart(
        query=inputs.get("query", ""),
        symbol=inputs.get("symbol", ""),
        chart_type=inputs.get("chart_type", ""),
        range_=inputs.get("range") or inputs.get("range_", "1d"),
        interval=inputs.get("interval", "5m"),
        title=inputs.get("title", ""),
        description=inputs.get("description", ""),
    )
    json_str = payload if isinstance(payload, str) else json.dumps(payload or {})
    return {"json": json_str, "content_type": "application/json"}


def run_chat(inputs: Dict[str, Any], memory: Dict[str, Any] | None = None) -> Dict[str, Any]:
    state: BaseState = mk_state("chat", inputs, mem_key=inputs.get("chat_id") or inputs.get("user_id"))
    if not is_graph_enabled("chat"):
        from app.agents.super.runner import run_super_agent as legacy_run
        mem = memory if memory is not None else {}
        out = legacy_run(inputs.get("query", ""), memory=mem)
        return {"response": out.get("output") or ""}
    saver = default_saver()
    final = chat_pipeline.run(state, saver=saver)
    response = (final.get("outputs") or {}).get("response") or ""
    return {"response": response}


def run_companies(inputs: Dict[str, Any]) -> Dict[str, Any]:
    state: BaseState = mk_state("companies", inputs, mem_key=None)
    if not is_graph_enabled("companies"):
        # Legacy path mirrors route code
        from supabase import create_client
        from app.tools.config import settings
        from app.services.scrapers.companies_scraper import get_companies
        import psycopg

        limit = inputs.get("limit")
        mode = (inputs.get("mode") or "upsert").lower()
        companies = get_companies() or []
        if limit is not None:
            companies = companies[: max(0, int(limit))]
        if not companies:
            return {"ingested": 0, "mode": mode}
        supa = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)
        payload = [{
            "company_name": c.get("company_name"),
            "nse_symbol": c.get("nse_symbol"),
            "bse_code": c.get("bse_code"),
            "bse_symbol": c.get("bse_symbol"),
            "isin": c.get("isin"),
            "industry": c.get("industry"),
            "status": c.get("status"),
            "market_cap": c.get("market_cap"),
        } for c in companies if c.get("isin")]
        if not payload:
            return {"ingested": 0, "mode": mode}
        CHUNK = 1000
        def _truncate():
            dsn = getattr(settings, "DATABASE_URL", None)
            if dsn:
                with psycopg.connect(dsn) as conn:
                    with conn.cursor() as cur:
                        cur.execute("TRUNCATE TABLE companies CASCADE;")
                        conn.commit()
            else:
                supa.table("companies").delete().neq("id", "").execute()
        def _fetch_isins():
            keep = set()
            start, step = 0, 1000
            while True:
                resp = supa.table("companies").select("isin").range(start, start+step-1).execute()
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
        def _delete_missing(keep):
            existing = _fetch_isins()
            to_delete = [v for v in existing if v not in keep]
            deleted = 0
            for i in range(0, len(to_delete), CHUNK):
                batch = to_delete[i:i+CHUNK]
                if not batch:
                    continue
                supa.table("companies").delete().in_("isin", batch).execute()
                deleted += len(batch)
            supa.table("companies").delete().is_("isin", None).execute()
            return deleted
        if mode == "truncate":
            _truncate()
            total = 0
            for i in range(0, len(payload), CHUNK):
                batch = payload[i:i+CHUNK]
                supa.table("companies").insert(batch).execute()
                total += len(batch)
            return {"ingested": total, "mode": mode}
        if mode == "replace":
            total = 0
            for i in range(0, len(payload), CHUNK):
                batch = payload[i:i+CHUNK]
                supa.table("companies").upsert(batch, on_conflict="isin").execute()
                total += len(batch)
            deleted = _delete_missing({p["isin"] for p in payload})
            return {"ingested": total, "deleted_missing": deleted, "mode": mode}
        total = 0
        for i in range(0, len(payload), CHUNK):
            batch = payload[i:i+CHUNK]
            supa.table("companies").upsert(batch, on_conflict="isin").execute()
            total += len(batch)
        return {"ingested": total, "mode": mode}

    saver = default_saver()
    final = companies_pipeline.run(state, saver=saver)
    resp = (final.get("outputs") or {}).get("response") or {}
    return resp
