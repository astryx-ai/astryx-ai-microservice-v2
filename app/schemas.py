from pydantic import BaseModel
from typing import Any, Optional


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
