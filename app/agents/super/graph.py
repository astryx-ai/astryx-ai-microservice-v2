from __future__ import annotations
from langgraph.graph import END, StateGraph
from typing import List, Dict, Any
from functools import partial

from .state import AgentState
from .resolver import resolve_ticker_node
from .stock import get_stock_node
from .news import get_news_node
from .formatting import merge_results_node
from .intents import classify_multi_intent, wants_expand_only, llm_tool_router
from .expand_news import expand_news_tool
from app.interfaces.local.client import (
    call_chart_local as call_chart_via_grpc,
    call_stock_local as call_stock_via_grpc_or_local,
    call_news_local as call_news_via_grpc_or_local,
)
from .company_extractor import _regex_extract as _regex_extract_candidates, extract_company as extract_company_pipeline
from .resolver import _load_company_db_subset


def _classify_node(state: AgentState) -> AgentState:
    q = state.get("question", "")
    memory = state.get("memory", {})
    intents = classify_multi_intent(q, memory)
    # Capture LLM entities like timeframe for charts, companies for later resolution
    try:
        route = llm_tool_router(q, memory)
        ents = (route or {}).get("entities") or {}
        if ents.get("timeframe"):
            state.setdefault("extras", {}).setdefault("chart_prefs", {})["timeframe"] = ents.get("timeframe")
        if ents.get("companies"):
            # hint for resolver
            state.setdefault("extras", {}).setdefault("entities", {})["companies"] = ents.get("companies")
    except Exception:
        pass
    state["intents"] = intents  # type: ignore
    return state


def _maybe_expand_news_node(state: AgentState) -> AgentState:
    intents: List[str] = state.get("intents", [])  # type: ignore
    if not intents or "expand_news" not in intents:
        return state
    # expand using last link in memory or provided param
    mem = state.get("memory", {})
    # pull context from memory when missing
    state.setdefault("company", mem.get("company"))
    state.setdefault("ticker", mem.get("ticker"))
    state.setdefault("exchange", mem.get("exchange"))
    link = mem.get("last_news_link") or mem.get("news_url") or ""
    # If no last link, try fetching latest news now and pick first link
    if not link:
        items = call_news_via_grpc_or_local(ticker=state.get("ticker"), company=state.get("company"))
        if items:
            link = items[0].get("url") or ""
            if link:
                mem["last_news_link"] = link
                state["memory"] = mem
        if not link:
            return state
    res = expand_news_tool(url=link, company=state.get("company"), ticker=state.get("ticker"))
    # Put a minimal news_items to render
    title = f"{state.get('company') or state.get('ticker') or 'Market'} — Expanded Summary"
    state["news_items"] = [{"title": title, "url": res.get("link",""), "summary": res.get("summary","")}]  # type: ignore
    try:
        mem["last_news_link"] = res.get("link") or link
        state["memory"] = mem
    except Exception:
        pass
    # Ensure intent includes news for downstream formatting
    intents = list(set(intents) | {"news"})
    state["intents"] = intents  # type: ignore
    return state


def _grpc_tools_node(state: AgentState) -> AgentState:
    """Run stock/news/chart tools via gRPC (or local fallback) based on intents.

    Executes stock and news in a simple parallel manner using asyncio run_in_executor pattern.
    """
    intents: List[str] = state.get("intents", [])  # type: ignore
    if not intents:
        return state
    q = state.get("question", "") or ""
    ticker = state.get("ticker")
    exchange = state.get("exchange")
    company = state.get("company")
    mem = state.get("memory", {}) or {}
    # Fallback to memory when current state lacks context
    mt, me, mc = mem.get("ticker"), mem.get("exchange"), mem.get("company")
    ticker = ticker or mt
    exchange = exchange or me
    company = company or mc
    # Also write back inferred context for downstream nodes/formatting
    if ticker and not state.get("ticker"):
        state["ticker"] = ticker
    if exchange and not state.get("exchange"):
        state["exchange"] = exchange
    if company and not state.get("company"):
        state["company"] = company

    async def _run_parallel_single(ticker, exchange, company) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        tasks = []
        out: Dict[str, Any] = {}
        if "stock" in intents:
            tasks.append(loop.run_in_executor(None, partial(call_stock_via_grpc_or_local, ticker=ticker, exchange=exchange, company=company)))
        else:
            tasks.append(None)
        # Fetch news when explicitly requested OR when chart is requested (news as well)
        if ("news" in intents) or ("chart" in intents):
            tasks.append(loop.run_in_executor(None, partial(call_news_via_grpc_or_local, ticker=ticker, company=company)))
        else:
            tasks.append(None)
        # Fetch chart only when chart intent is present
        if "chart" in intents:
            symbol = state.get("full_symbol") or (
                f"{(ticker or '').upper()}.NS" if (ticker and (exchange=="NSE")) else (
                    f"{(ticker or '').upper()}.BO" if (ticker and (exchange=="BSE")) else (ticker or "")
                )
            )
            # Pull timeframe prefs if any
            cp = (state.get("extras", {}) or {}).get("chart_prefs", {})
            tf = (cp.get("timeframe") or "").lower()
            # Parse basic forms like "1d 5m" or "5y 1d"
            rng, intr = "1d", "5m"
            try:
                import re as _re
                m = _re.search(r"\b(\d+[dwmy])\s+(\d+(?:m|d|wk|mo))\b", tf)
                if m:
                    rng, intr = m.group(1), m.group(2)
            except Exception:
                pass
            # Provide chat_id for trace/debug if present in memory
            chat_id = (state.get("memory", {}) or {}).get("_chat_id") or ""
            tasks.append(loop.run_in_executor(None, partial(call_chart_via_grpc, query=q, symbol=symbol, range_=rng, interval=intr, chat_id=chat_id)))
        else:
            tasks.append(None)

        results = []
        for t in tasks:
            if t is None:
                results.append(None)
            else:
                results.append(await t)
        return {"stock": results[0], "news": results[1], "chart": results[2]}

    try:
        import asyncio
        matches = state.get("matches") or []
        # Fallback: derive multi matches from query if resolver didn't set them
        if not matches:
            try:
                cands = _regex_extract_candidates(q)
            except Exception:
                cands = []
            if len(cands) >= 2:
                subset = _load_company_db_subset(q)
                uniq = {}
                for c in cands[:5]:
                    sub = extract_company_pipeline(c, subset) or []
                    if not sub:
                        continue
                    top = max(sub, key=lambda x: int(x.get("confidence") or 0))
                    key = (top.get("name") or "").strip().lower()
                    if key and key not in uniq:
                        uniq[key] = top
                matches = list(uniq.values())
                if matches:
                    state["matches"] = matches  # type: ignore
        multi_results = []
        multi_charts = []
        if matches and len(matches) > 1:
            # Query each matched company (cap to 5 to avoid over-fetch)
            limit = 5
            async def gather_all():
                tasks = []
                for m in matches[:limit]:
                    m_ticker = m.get("nse") or m.get("bse")
                    m_ex = "NSE" if m.get("nse") else ("BSE" if m.get("bse") else None)
                    m_company = m.get("name")
                    tasks.append(_run_parallel_single(m_ticker, m_ex, m_company))
                return await asyncio.gather(*tasks)
            results_list = asyncio.run(gather_all())
            for m, r in zip(matches[:limit], results_list):
                multi_results.append({
                    "company": m.get("name"),
                    "ticker": (m.get("nse") or m.get("bse")),
                    "exchange": ("NSE" if m.get("nse") else ("BSE" if m.get("bse") else None)),
                    "stock_data": r.get("stock"),
                    "news_items": r.get("news"),
                })
                if r.get("chart"):
                    multi_charts.append({
                        "company": m.get("name"),
                        "symbol": ((m.get("nse") or m.get("bse") or "").upper() + (".NS" if m.get("nse") else (".BO" if m.get("bse") else ""))),
                        "payload": r.get("chart"),
                    })
            # Stash first chart only to avoid ambiguity
            if results_list and results_list[0].get("chart"):
                chart = results_list[0]["chart"] or {}
                meta = chart.get("meta") or {}
                if not meta.get("symbol"):
                    sym = state.get("full_symbol") or (
                        f"{(state.get('ticker') or '').upper()}.NS" if state.get("exchange") == "NSE" else (
                            f"{(state.get('ticker') or '').upper()}.BO" if state.get("exchange") == "BSE" else (state.get('ticker') or '').upper()
                        )
                    )
                    if sym:
                        meta["symbol"] = sym
                        chart["meta"] = meta
                ex = state.setdefault("extras", {})
                ex["chart"] = chart  # type: ignore
                if multi_charts:
                    ex["charts"] = multi_charts  # type: ignore
            state["multi_results"] = multi_results  # type: ignore
        else:
            results = asyncio.run(_run_parallel_single(ticker, exchange, company))
    except RuntimeError:
        loop = asyncio.get_event_loop()
        matches = state.get("matches") or []
        multi_results = []
        multi_charts = []
        if matches and len(matches) > 1:
            limit = 5
            def run_blocking(m):
                m_t = m.get("nse") or m.get("bse")
                m_e = "NSE" if m.get("nse") else ("BSE" if m.get("bse") else None)
                m_c = m.get("name")
                coro = _run_parallel_single(m_t, m_e, m_c)
                return loop.run_until_complete(coro)
            results_list = [run_blocking(m) for m in matches[:limit]]
            for m, r in zip(matches[:limit], results_list):
                multi_results.append({
                    "company": m.get("name"),
                    "ticker": (m.get("nse") or m.get("bse")),
                    "exchange": ("NSE" if m.get("nse") else ("BSE" if m.get("bse") else None)),
                    "stock_data": r.get("stock"),
                    "news_items": r.get("news"),
                })
                if r.get("chart"):
                    multi_charts.append({
                        "company": m.get("name"),
                        "symbol": ((m.get("nse") or m.get("bse") or "").upper() + (".NS" if m.get("nse") else (".BO" if m.get("bse") else ""))),
                        "payload": r.get("chart"),
                    })
            if results_list and results_list[0].get("chart"):
                chart = results_list[0]["chart"] or {}
                meta = chart.get("meta") or {}
                if not meta.get("symbol"):
                    sym = state.get("full_symbol") or (
                        f"{(state.get('ticker') or '').upper()}.NS" if state.get("exchange") == "NSE" else (
                            f"{(state.get('ticker') or '').upper()}.BO" if state.get("exchange") == "BSE" else (state.get('ticker') or '').upper()
                        )
                    )
                    if sym:
                        meta["symbol"] = sym
                        chart["meta"] = meta
                ex = state.setdefault("extras", {})
                ex["chart"] = chart  # type: ignore
                if multi_charts:
                    ex["charts"] = multi_charts  # type: ignore
            state["multi_results"] = multi_results  # type: ignore
        else:
            results = loop.run_until_complete(_run_parallel_single(ticker, exchange, company))  # type: ignore

    # Populate single-result fields for backward compatibility
    if 'results' in locals() and isinstance(results, dict):
        if results.get("stock"):
            state["stock_data"] = results["stock"]  # type: ignore
        if results.get("news"):
            state["news_items"] = results["news"]  # type: ignore
            # Persist the most recent news link into memory to support "elaborate more"
            try:
                mem = state.get("memory", {}) or {}
                items = results.get("news") or []
                if isinstance(items, list) and items:
                    link = items[0].get("url") or items[0].get("source") or ""
                    if link:
                        mem["last_news_link"] = link
                        state["memory"] = mem
            except Exception:
                pass
        if results.get("chart"):
            chart = results["chart"] or {}
            meta = chart.get("meta") or {}
            if not meta.get("symbol"):
                sym = state.get("full_symbol") or (
                    f"{(state.get('ticker') or '').upper()}.NS" if state.get("exchange") == "NSE" else (
                        f"{(state.get('ticker') or '').upper()}.BO" if state.get("exchange") == "BSE" else (state.get('ticker') or '').upper()
                    )
                )
                if sym:
                    meta["symbol"] = sym
                    chart["meta"] = meta
            state.setdefault("extras", {})["chart"] = chart  # type: ignore
    return state


def build_super_agent():
    sg = StateGraph(AgentState)
    sg.add_node("classify", _classify_node)
    sg.add_node("resolve_ticker", resolve_ticker_node)
    sg.add_node("maybe_expand", _maybe_expand_news_node)
    sg.add_node("grpc_tools", _grpc_tools_node)
    sg.add_node("get_stock", get_stock_node)  # keep local as fallback path
    sg.add_node("get_news", get_news_node)    # keep local as fallback path
    sg.add_node("merge_results", merge_results_node)

    sg.set_entry_point("classify")

    def after_classify(state: AgentState):
        intents: List[str] = state.get("intents", [])  # type: ignore
        if wants_expand_only(intents):
            return "maybe_expand"
        # For any stock/news/chart need, resolve first
        if {"stock","news","chart"} & set(intents):
            return "resolve_ticker"
        # Else, directly merge (casual)
        return "merge_results"

    sg.add_conditional_edges("classify", after_classify, {"maybe_expand": "maybe_expand", "resolve_ticker": "resolve_ticker", "merge_results": "merge_results"})

    def after_resolve(state: AgentState):
        return "grpc_tools"

    sg.add_edge("resolve_ticker", "grpc_tools")
    sg.add_edge("maybe_expand", "merge_results")
    sg.add_edge("grpc_tools", "merge_results")
    sg.add_edge("merge_results", END)
    return sg.compile()
