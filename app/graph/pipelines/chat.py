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
    """No-op loader when using LangGraph checkpointer.

    With a configured checkpointer and a stable thread_id, LangGraph restores the
    prior state automatically. We keep this node for timing/structure symmetry.
    """
    return state


@_node("super_agent")
def super_agent_node(state: BaseState) -> BaseState:
    from app.agents.super.runner import run_super_agent
    q = (state.get("inputs", {}).get("query") or "").strip()
    mem = state.get("memory", {})
    with timed(state.setdefault("metrics", {}).setdefault("node_durations", {}), "super_agent"):
        thread_id = state.get("inputs", {}).get("chat_id") or state.get("inputs", {}).get("user_id")
        result: Dict[str, Any] = run_super_agent(q, memory=mem, thread_id=thread_id)
    state.setdefault("outputs", {})["response"] = result.get("output") or ""
    # Bubble up chart JSON (if available) into outputs
    extras = (result or {}).get("extras") or {}
    chart_payload = extras.get("chart")
    chart_multi = extras.get("charts")
    if chart_payload:
        state.setdefault("outputs", {})["chart"] = chart_payload
    if chart_multi:
        state.setdefault("outputs", {})["charts"] = chart_multi
    # Persist updated memory from the super agent run into this pipeline state
    state["memory"] = result.get("memory", mem)
    return state


@_node("save_memory")
def save_memory_node(state: BaseState) -> BaseState:
    """No-op saver when using LangGraph checkpointer.

    The checkpointer persists the state at each step keyed by thread_id.
    """
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
    # Use chat_id or user_id as the persistent thread identifier.
    inputs = state.get("inputs", {})
    thread_id = inputs.get("chat_id") or inputs.get("user_id") or "anonymous"
    config = {"configurable": {"thread_id": thread_id}}
    if saver is not None:
        config["checkpointer"] = saver
    return graph.invoke(state, config=config)
