from typing import TypedDict, List, Optional, Any


class AgentState(TypedDict, total=False):
    messages: List[Any]
    user_id: Optional[str]
    chat_id: Optional[str]


