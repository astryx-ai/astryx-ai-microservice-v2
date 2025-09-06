from pydantic import BaseModel, Field
from typing import List, Literal, Optional


class Candle(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None
    timestamp: str


class CandleSeries(BaseModel):
    instrument_key: str
    unit: Literal['minutes', 'hours', 'days', 'weeks', 'months']
    interval: str
    from_date: str
    to_date: str
    candles: List[Candle] = Field(default_factory=list)
    provider: str


class CandleQuery(BaseModel):
    instrument_key: str
    unit: Literal['minutes', 'hours', 'days', 'weeks', 'months']
    interval: str
    from_date: str
    to_date: str 