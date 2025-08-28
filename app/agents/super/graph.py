from __future__ import annotations
from langgraph.graph import END, StateGraph

from .state import AgentState
from .resolver import resolve_ticker_node
from .stock import get_stock_node
from .news import get_news_node
from .formatting import merge_results_node


def build_super_agent():
    sg = StateGraph(AgentState)
    sg.add_node("resolve_ticker", resolve_ticker_node)
    sg.add_node("get_stock", get_stock_node)
    sg.add_node("get_news", get_news_node)
    sg.add_node("merge_results", merge_results_node)

    sg.set_entry_point("resolve_ticker")

    def after_resolve(state: AgentState):
        intent = state.get("intent", "both")
        if intent == "stock":
            return "get_stock"
        if intent == "news":
            return "get_news"
        if intent == "both":
            return "get_stock"
        # greeting or clarify -> go directly to merge (will render friendly/clarify message)
        return "merge_results"

    sg.add_conditional_edges(
        "resolve_ticker", after_resolve, {"get_stock": "get_stock", "get_news": "get_news", "merge_results": "merge_results"}
    )

    def after_stock(state: AgentState):
        return "get_news" if state.get("intent") == "both" else "merge_results"

    sg.add_conditional_edges("get_stock", after_stock, {"get_news": "get_news", "merge_results": "merge_results"})
    sg.add_edge("get_news", "merge_results")
    sg.add_edge("merge_results", END)
    return sg.compile()
