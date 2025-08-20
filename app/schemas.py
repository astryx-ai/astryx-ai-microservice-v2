from pydantic import BaseModel
from typing import Any, Optional, List, Dict


class AIChatRequest(BaseModel):
    query: str
    user_id: str
    chat_id: str


class AIChatResponseData(BaseModel):
    response: str
    chart_data: Any | None = None
    tokens_used: int
    cost: float


class AIChatResponse(BaseModel):
    success: bool
    error: Optional[str] = None
    data: Optional[AIChatResponseData] = None


# RAG Ask
class AskRequest(BaseModel):
    query: str
    symbol: str
    user_id: Optional[str] = None
    chat_id: Optional[str] = None


class AskResponse(BaseModel):
    success: bool
    answer: Optional[str] = None
    sources: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None


# Candles
class CandleItem(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None


class CandleResponse(BaseModel):
    symbol: str
    days: int
    candles: List[CandleItem] | list
