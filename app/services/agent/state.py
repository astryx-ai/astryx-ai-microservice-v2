from typing import TypedDict, List, Optional, Any


class AgentState(TypedDict, total=False):
    messages: List[Any]
    user_id: Optional[str]
    chat_id: Optional[str]
    route: Optional[str]            # e.g., "standard", "deep_research", "chart_viz", "legal_analysis"
    decision_reason: Optional[str]
    # Optional convenience fields for routers
    query: Optional[str]
    context: Optional[List[Any]]

# Single source of truth for all available routes/subgraphs
AVAILABLE_ROUTES: List[str] = [
    "standard",
    "deep_research",
    "chart_viz",
    "financial_analysis",
]
