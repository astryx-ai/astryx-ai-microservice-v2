from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional, TypedDict
from typing_extensions import Annotated
from dataclasses import dataclass
from datetime import datetime


def _keep_first(a: Optional[str], b: Optional[str]) -> Optional[str]:
    """Reducer for LangGraph merges: keep the first non-null question value."""
    return a if a is not None else b


class AgentState(TypedDict, total=False):
    question: Annotated[Optional[str], _keep_first]
    company: Optional[str]
    ticker: Optional[str]
    exchange: Optional[Literal["NSE", "BSE"]]
    intent: Literal["stock", "news", "both", "greeting", "clarify"]
    intents: List[str]
    news_detail: Literal["short", "medium", "long"]
    stock_data: Optional[Dict[str, Any]]
    news_items: Optional[List[Dict[str, Any]]]
    # Multi-company support
    matches: Optional[List[Dict[str, Any]]]
    multi_results: Optional[List[Dict[str, Any]]]
    output: Optional[str]
    suggestions: Optional[List[Dict[str, Any]]]
    memory: Dict[str, Any]
    now: datetime
    extras: Dict[str, Any]


@dataclass
class TickerRecord:
    company_name: str
    nse_symbol: Optional[str]
    bse_symbol: Optional[str]
