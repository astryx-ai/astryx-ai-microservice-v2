from __future__ import annotations
from typing import Any, Dict, List, Set

try:
    from langgraph.graph import StateGraph, END
except Exception:  # pragma: no cover
    StateGraph = None  # type: ignore
    END = None  # type: ignore

from app.graph.state import BaseState
from app.graph.utils import timed


def _node(name):
    def deco(fn):
        fn._node_name = name  # type: ignore[attr-defined]
        return fn
    return deco


def _supabase():
    from supabase import create_client
    from app.tools.config import settings
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)


def _truncate_companies_table() -> None:
    from app.tools.config import settings
    dsn = getattr(settings, "DATABASE_URL", None)
    if dsn:
        import psycopg
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE companies CASCADE;")
                conn.commit()
        return
    supa = _supabase()
    supa.table("companies").delete().neq("id", "").execute()


def _fetch_all_isins(supa) -> Set[str]:
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


def _delete_missing_isins(supa, keep_isins: Set[str]) -> int:
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
    supa.table("companies").delete().is_("isin", None).execute()
    return deleted


@_node("fetch_companies")
def fetch_companies_node(state: BaseState) -> BaseState:
    from app.services.scrapers.companies_scraper import get_companies
    inputs = state.get("inputs", {})
    limit = inputs.get("limit")
    with timed(state.setdefault("metrics", {}).setdefault("node_durations", {}), "fetch_companies"):
        companies = get_companies() or []
        if isinstance(limit, int) and limit is not None:
            companies = companies[: max(0, int(limit))]
    state.setdefault("memory", {})["companies_raw"] = companies
    return state


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


@_node("prepare_payload")
def prepare_payload_node(state: BaseState) -> BaseState:
    items: List[Dict[str, Any]] = state.get("memory", {}).get("companies_raw", [])
    payload = [_coerce_record(x) for x in items if x.get("isin")]
    state.setdefault("memory", {})["companies_payload"] = payload
    return state


@_node("apply_mode")
def apply_mode_node(state: BaseState) -> BaseState:
    inputs = state.get("inputs", {})
    mode = (inputs.get("mode") or "upsert").lower()
    payload: List[Dict[str, Any]] = state.get("memory", {}).get("companies_payload", [])
    supa = _supabase()
    CHUNK = 1000
    out: Dict[str, Any] = {"mode": mode}
    with timed(state.setdefault("metrics", {}).setdefault("node_durations", {}), f"companies_{mode}"):
        if mode == "truncate":
            _truncate_companies_table()
            total = 0
            for i in range(0, len(payload), CHUNK):
                batch = payload[i : i + CHUNK]
                supa.table("companies").insert(batch).execute()
                total += len(batch)
            out["ingested"] = total
        elif mode == "replace":
            total = 0
            for i in range(0, len(payload), CHUNK):
                batch = payload[i : i + CHUNK]
                supa.table("companies").upsert(batch, on_conflict="isin").execute()
                total += len(batch)
            out["ingested"] = total
            out["deleted_missing"] = _delete_missing_isins(supa, {p["isin"] for p in payload})
        else:
            total = 0
            for i in range(0, len(payload), CHUNK):
                batch = payload[i : i + CHUNK]
                supa.table("companies").upsert(batch, on_conflict="isin").execute()
                total += len(batch)
            out["ingested"] = total
    state.setdefault("outputs", {})["counts"] = {"ingested": out.get("ingested", 0)}
    state.setdefault("outputs", {})["response"] = out
    return state


def build_graph():
    if StateGraph is None:
        return None
    sg = StateGraph(BaseState)  # type: ignore[type-arg]
    sg.add_node("fetch_companies", fetch_companies_node)
    sg.add_node("prepare_payload", prepare_payload_node)
    sg.add_node("apply_mode", apply_mode_node)
    sg.set_entry_point("fetch_companies")
    sg.add_edge("fetch_companies", "prepare_payload")
    sg.add_edge("prepare_payload", "apply_mode")
    sg.add_edge("apply_mode", END)
    return sg.compile()


def run(state: BaseState, saver=None) -> BaseState:
    graph = build_graph()
    if graph is None:
        s = fetch_companies_node(state)
        s = prepare_payload_node(s)
        s = apply_mode_node(s)
        return s
    if saver is not None:
        return graph.invoke(state, config={"checkpointer": saver})
    return graph.invoke(state)
