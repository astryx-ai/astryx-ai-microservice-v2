from __future__ import annotations
from typing import Any, Dict, List, Optional, TypedDict, Literal
from datetime import datetime


class ErrorRecord(TypedDict, total=False):
    node: str
    error: str
    ts: str
    retry: int


class Outputs(TypedDict, total=False):
    response: Optional[str]
    chart: Optional[Dict[str, Any]]
    docs: Optional[List[Any]]  # langchain Documents
    counts: Optional[Dict[str, int]]


class Inputs(TypedDict, total=False):
    query: Optional[str]
    symbol: Optional[str]
    range: Optional[str]
    interval: Optional[str]
    limit: Optional[int]
    user_id: Optional[str]
    chat_id: Optional[str]
    params: Optional[Dict[str, Any]]


class Metrics(TypedDict, total=False):
    node_durations: Dict[str, float]
    tokens_used: Optional[int]
    cost: Optional[float]


class Schedule(TypedDict, total=False):
    next_run_at: Optional[str]
    loop: Optional[bool]


class BaseState(TypedDict, total=False):
    run_id: str
    pipeline: Literal["news", "companies", "stocks", "chat", "chart"]
    started_at: str
    finished_at: Optional[str]
    status: Literal["pending", "running", "success", "partial_success", "failed"]
    inputs: Inputs
    outputs: Outputs
    memory: Dict[str, Any]
    mem_key: Optional[str]
    errors: List[ErrorRecord]
    metrics: Metrics
    schedule: Schedule


def mk_state(pipeline: BaseState["pipeline"], inputs: Inputs, mem_key: Optional[str] = None) -> BaseState:  # type: ignore[index]
    now = datetime.utcnow().isoformat()
    return BaseState(
        run_id=f"{pipeline}-{now}",
        pipeline=pipeline,
        started_at=now,
        status="running",
        inputs=inputs,
        outputs=Outputs(),
        memory={},
        mem_key=mem_key,
        errors=[],
        metrics={"node_durations": {}},
        schedule={},
    )
