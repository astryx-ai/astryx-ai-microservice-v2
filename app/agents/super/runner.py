from __future__ import annotations
from typing import Dict, Optional
from datetime import datetime, timezone
from app.graph.checkpoints import default_saver
from app.tools.memory_store import global_memory_store
from app.tools.config import settings


def run_super_agent(question: str, memory: Optional[Dict[str, any]] = None, thread_id: Optional[str] = None) -> Dict[str, any]:
    from .graph import build_super_agent
    from .state import AgentState

    graph = build_super_agent()
    now = datetime.now(timezone.utc)
    # Load persisted memory if thread_id provided
    store = global_memory_store()
    ttl = None
    try:
        ttl_val = getattr(settings, "MEMORY_TTL_SECONDS", 0)
        # Treat non-positive as disabled TTL (persist across session)
        ttl = int(ttl_val) if int(ttl_val or 0) > 0 else None
    except Exception:
        ttl = None
    persisted = (store.load(chat_id=thread_id, user_id=None, ttl_seconds=ttl) if thread_id else {})
    base_mem = {}
    base_mem.update(persisted or {})
    base_mem.update(memory or {})
    init: AgentState = {"question": question, "memory": base_mem, "now": now}
    # Use LangGraph checkpointer if a thread_id is provided (e.g., chat_id)
    saver = default_saver()
    if thread_id and saver is not None:
        config = {"checkpointer": saver, "configurable": {"thread_id": thread_id}}
        final_state: AgentState = graph.invoke(init, config=config)
    else:
        final_state = graph.invoke(init)
    # Save memory back
    try:
        mem = final_state.get("memory", base_mem)
        if thread_id:
            store.save(chat_id=thread_id, user_id=None, memory=mem)
    except Exception:
        pass
    return final_state


def resolve_company_ticker(question: str, memory: Optional[Dict[str, any]] = None) -> Dict[str, Optional[str]]:
    from .resolver import resolve_ticker_node
    from .state import AgentState

    now = datetime.now(timezone.utc)
    state: AgentState = {"question": question, "memory": memory or {}, "now": now}
    state = resolve_ticker_node(state)
    mem = state.get("memory", memory or {})
    return {
        "company": state.get("company"),
        "ticker": state.get("ticker"),
        "exchange": state.get("exchange"),
        "memory": mem,
    }


def resolve_company_ticker_fast(query: str) -> Dict[str, Optional[str]]:
    import re
    from .resolver import _normalize_company_query, _fuzzy_match_company, _supabase_client
    from .utils import yahoo_search_symbol

    s = (query or "").strip()
    m = re.search(r"\bfor\s+([A-Za-z0-9&\./\-\s]{2,60})", s, re.I)
    phrase = m.group(1).strip() if m else None
    if not phrase:
        qs = re.findall(r"[\"']([^\"']+)[\"']", s)
        phrase = (qs[-1].strip() if qs else s)
    phrase = re.sub(r"\s+(?:on|in|with|using|chart|line|area|bar|candle|candlestick|ohlc)\b.*$", "", phrase, flags=re.I).strip()
    if "." in phrase or phrase.startswith("^"):
        sym = phrase.upper()
        if sym.endswith(".NS"):
            return {"company": None, "ticker": sym[:-3], "exchange": "NSE"}
        if sym.endswith(".BO"):
            return {"company": None, "ticker": sym[:-3], "exchange": "BSE"}
        return {"company": None, "ticker": sym, "exchange": None}

    sb = _supabase_client()
    row = None
    if sb:
        try:
            qnorm = _normalize_company_query(phrase)
            upper_q = qnorm.upper()
            is_short_upper = len(upper_q) <= 6 and upper_q == phrase.upper()

            # New: Exact symbol match first for likely tickers
            if is_short_upper:
                exact = sb.table("companies").select("company_name,nse_symbol,bse_symbol").or_(
                    f"nse_symbol.eq.{upper_q},bse_symbol.eq.{upper_q}"
                ).limit(1).execute()
                if exact.data:
                    row = exact.data[0]

            if not row:
                toks = [t for t in re.split(r"\W+", qnorm) if t]
                clauses = [f"company_name.ilike.%{t}%" for t in toks[:3]] or [f"company_name.ilike.%{qnorm}%"]
                res = sb.table("companies").select("company_name,nse_symbol,bse_symbol").or_(",".join(clauses)).limit(100).execute()
                rows = res.data or []
                row = _fuzzy_match_company(phrase, rows)  # Uses updated fuzzy with short-query tightening
        except Exception:
            row = None

    if row:
        ticker = row.get("nse_symbol") or row.get("bse_symbol")
        exch: Optional[str] = "NSE" if row.get("nse_symbol") else ("BSE" if row.get("bse_symbol") else None)
        return {"company": row.get("company_name"), "ticker": ticker, "exchange": exch}

    # Yahoo as final fallback
    y = yahoo_search_symbol(phrase)
    if y and y.get("ticker"):
        return {"company": y.get("company"), "ticker": y.get("ticker"), "exchange": y.get("exchange")}

    return {"company": None, "ticker": None, "exchange": None}