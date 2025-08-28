from __future__ import annotations
from typing import Any, Dict, List

try:
    from langgraph.graph import StateGraph, END
except Exception:  # pragma: no cover
    StateGraph = None  # type: ignore
    END = None  # type: ignore

from app.graph.state import BaseState
from app.graph.utils import timed, network_retry

# Lazy imports inside nodes to avoid hard dependency at module import time


def _node(name):
    def deco(fn):
        fn._node_name = name  # type: ignore[attr-defined]
        return fn
    return deco


@_node("fetch_news")
def fetch_news_node(state: BaseState) -> BaseState:
    inputs = state.get("inputs", {})
    company = (inputs.get("company") or inputs.get("query") or inputs.get("ticker") or inputs.get("symbol") or "").strip()
    limit = int(inputs.get("limit") or 20)
    with timed(state.setdefault("metrics", {}).setdefault("node_durations", {}), "fetch_news"):
        from app.services.scrapers.scrape_news import get_news
        items = get_news(company, limit=limit) or []
    state.setdefault("memory", {})["news_items"] = items
    return state


@_node("prepare_docs")
def prepare_docs_node(state: BaseState) -> BaseState:
    inputs = state.get("inputs", {})
    ticker = (inputs.get("ticker") or inputs.get("symbol") or inputs.get("query") or "").strip()
    items: List[Dict[str, Any]] = state.get("memory", {}).get("news_items", [])
    docs = []
    with timed(state.setdefault("metrics", {}).setdefault("node_durations", {}), "prepare_docs"):
        from app.services.scrapers.sanitize import clean_text
        from app.tools.rag import chunk_text
        for it in items:
            title = (it.get("title") or "").strip()
            text = (it.get("text") or it.get("summary") or "").strip()
            blob = clean_text(f"{title}. {text}")
            if not blob:
                continue
            meta: Dict[str, Any] = {
                "ticker": (inputs.get("ticker") or inputs.get("symbol") or "").upper(),
                "company": (inputs.get("company") or ""),
                "source": (it.get("url") or it.get("source") or "news"),
                "title": title,
                "type": "news",
            }
            docs.extend(chunk_text(blob, meta))
    state.setdefault("outputs", {})["docs"] = docs
    state.setdefault("outputs", {}).setdefault("counts", {})["docs"] = len(docs)
    return state


@_node("upsert_news")
def upsert_news_node(state: BaseState) -> BaseState:
    docs = (state.get("outputs", {}) or {}).get("docs") or []
    with timed(state.setdefault("metrics", {}).setdefault("node_durations", {}), "upsert_news"):
        if docs:
            from app.tools.rag import upsert_news
            upsert_news(docs)
    return state


def build_graph():
    if StateGraph is None:
        return None
    sg = StateGraph(BaseState)  # type: ignore[type-arg]
    sg.add_node("fetch_news", fetch_news_node)
    sg.add_node("prepare_docs", prepare_docs_node)
    sg.add_node("upsert_news", upsert_news_node)
    sg.set_entry_point("fetch_news")
    sg.add_edge("fetch_news", "prepare_docs")
    sg.add_edge("prepare_docs", "upsert_news")
    sg.add_edge("upsert_news", END)
    return sg.compile()


def run(state: BaseState, saver=None) -> BaseState:
    """Execute the news pipeline. If LangGraph isn't available, fall back inline.

    Returns the final state with outputs.counts.docs set, matching existing route response.
    """
    graph = build_graph()
    if graph is None:
        # Fallback: sequential execution without graph
        s = fetch_news_node(state)
        s = prepare_docs_node(s)
        s = upsert_news_node(s)
        return s
    if saver is not None:
        return graph.invoke(state, config={"checkpointer": saver})
    return graph.invoke(state)
