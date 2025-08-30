from __future__ import annotations
from typing import Any, Dict, List, Optional

# This module provides a minimal agent wrapper for HTTP streaming routes.
# It intentionally has no gRPC dependencies. It adapts the existing super agent
# to an interface with invoke(...), and stub async streaming methods can be
# added later if needed.


class _SimpleAgent:
    def __init__(self) -> None:
        pass

    # Optional async streaming APIs are intentionally omitted. The HTTP route
    # will detect their absence and fall back to invoke(...).

    def invoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous invoke that returns a dict with a messages list.

        Input payload is expected to contain:
          - messages: List of objects (may be langchain messages or simple dicts)
          - thread_id (optional): to persist memory across turns
        """
        from app.agents.super.runner import run_super_agent

        messages: List[Any] = list(payload.get("messages") or [])
        # Extract the latest user input text from provided messages
        question: str = ""
        for m in reversed(messages):
            # Try to use the last Human message; otherwise any content field
            m_type = getattr(m, "type", None) or (m.get("type") if isinstance(m, dict) else None)
            content = getattr(m, "content", None)
            if content is None and isinstance(m, dict):
                content = m.get("content")
            if not question and (m_type == "human" or content):
                question = str(content or "").strip()
                if m_type == "human":
                    break

        thread_id: Optional[str] = payload.get("thread_id")
        result: Dict[str, Any] = run_super_agent(question, memory={}, thread_id=thread_id)
        text = (result.get("output") if isinstance(result, dict) else None) or ""
        # Provide a minimal messages-like structure for downstream parsing
        return {
            "messages": [
                {"type": "ai", "content": text}
            ]
        }


def build_agent() -> _SimpleAgent:
    """Return a minimal agent compatible with the HTTP streaming route.

    No gRPC imports. Streaming methods are optional and not required.
    """
    return _SimpleAgent()


def agent_answer(question: str) -> str:
    """Convenience helper used by the non-streaming /agent shim.

    It uses the existing super agent and returns the final text answer.
    """
    from app.agents.super.runner import run_super_agent

    result: Dict[str, Any] = run_super_agent(question, memory={})
    return (result.get("output") if isinstance(result, dict) else None) or ""
