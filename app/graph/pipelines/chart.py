from __future__ import annotations
from typing import Any, Dict
import json
import asyncio

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


def _normalize_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "query": (inputs.get("query") or "").strip(),
        "symbol": (inputs.get("symbol") or "").strip(),
        "chart_type": (inputs.get("chart_type") or "").strip(),
        "range_": (inputs.get("range") or inputs.get("range_") or "1d"),
        "interval": (inputs.get("interval") or "5m"),
        "title": (inputs.get("title") or "").strip(),
        "description": (inputs.get("description") or "").strip(),
    }


@_node("normalize")
def normalize_node(state: BaseState) -> BaseState:
    inputs = state.get("inputs", {})
    norm = _normalize_inputs(inputs)
    state.setdefault("memory", {})["norm"] = norm
    return state


async def _call_build_chart(norm: Dict[str, Any]) -> Dict[str, Any]:
    from app.features.charts.builder import build_chart
    # build_chart is async in our codebase; call accordingly
    return await build_chart(**norm)


@_node("build_chart")
def build_chart_node(state: BaseState) -> BaseState:
    norm = state.get("memory", {}).get("norm") or {}
    with timed(state.setdefault("metrics", {}).setdefault("node_durations", {}), "build_chart"):
        try:
            payload = asyncio.run(_call_build_chart(norm))
        except RuntimeError:
            # Already in an event loop (e.g., FastAPI); run with create_task style
            payload = asyncio.get_event_loop().run_until_complete(_call_build_chart(norm))  # type: ignore
    state.setdefault("outputs", {})["chart"] = payload or {}
    return state


def build_graph():
    if StateGraph is None:
        return None
    sg = StateGraph(BaseState)  # type: ignore[type-arg]
    sg.add_node("normalize", normalize_node)
    sg.add_node("build_chart", build_chart_node)
    sg.set_entry_point("normalize")
    sg.add_edge("normalize", "build_chart")
    sg.add_edge("build_chart", END)
    return sg.compile()


def run(state: BaseState, saver=None) -> BaseState:
    graph = build_graph()
    if graph is None:
        s = normalize_node(state)
        s = build_chart_node(s)
        return s
    if saver is not None:
        return graph.invoke(state, config={"checkpointer": saver})
    return graph.invoke(state)
