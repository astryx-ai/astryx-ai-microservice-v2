from __future__ import annotations
from typing import Any

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except Exception:  # pragma: no cover
    SqliteSaver = None  # type: ignore


def default_saver(db_path: str = ".run/graph.db"):
    """Return a SqliteSaver if available; else a no-op saver (None).

    We keep this internal; callers must handle None by running without persistence.
    """
    if SqliteSaver is None:
        return None
    # Use file path relative to workspace; create parent dirs if needed
    import os
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return SqliteSaver(db_path)
