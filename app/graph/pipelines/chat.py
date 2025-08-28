from __future__ import annotations
from typing import Any, Dict

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


@_node("load_memory")
def load_memory_node(state: BaseState) -> BaseState:
    from app.tools.memory_store import global_memory_store
    key = state.get("mem_key") or (state.get("inputs", {}).get("chat_id") or state.get("inputs", {}).get("user_id"))
    store = global_memory_store()
    mem = store.load(state.get("inputs", {}).get("chat_id"), state.get("inputs", {}).get("user_id"), ttl_seconds=900)
    state["memory"] = mem or {}
    state["mem_key"] = key
    return state


@_node("super_agent")
def super_agent_node(state: BaseState) -> BaseState:
    from app.agents.super.runner import run_super_agent
    q = (state.get("inputs", {}).get("query") or "").strip()
    mem = state.get("memory", {})
    with timed(state.setdefault("metrics", {}).setdefault("node_durations", {}), "super_agent"):
        result: Dict[str, Any] = run_super_agent(q, memory=mem)
    state.setdefault("outputs", {})["response"] = result.get("output") or ""
    state["memory"] = mem
    return state


@_node("save_memory")
def save_memory_node(state: BaseState) -> BaseState:
    from app.tools.memory_store import global_memory_store
    store = global_memory_store()
    mem = state.get("memory", {})
    store.save(state.get("inputs", {}).get("chat_id"), state.get("inputs", {}).get("user_id"), mem)
    return state


def build_graph():
    if StateGraph is None:
        return None
    sg = StateGraph(BaseState)  # type: ignore[type-arg]
    sg.add_node("load_memory", load_memory_node)
    sg.add_node("super_agent", super_agent_node)
    sg.add_node("save_memory", save_memory_node)
    sg.set_entry_point("load_memory")
    sg.add_edge("load_memory", "super_agent")
    sg.add_edge("super_agent", "save_memory")
    sg.add_edge("save_memory", END)
    return sg.compile()


def run(state: BaseState, saver=None) -> BaseState:
    graph = build_graph()
    if graph is None:
        s = load_memory_node(state)
        s = super_agent_node(s)
        s = save_memory_node(s)
        return s
    if saver is not None:
        return graph.invoke(state, config={"checkpointer": saver})
    return graph.invoke(state)
