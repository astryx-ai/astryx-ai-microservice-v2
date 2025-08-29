"""Deprecated shim module for /agent.

The actual shim route is defined in app.routes.chat (agent_router).
We re-export its router here to preserve imports.
"""

from app.routes.chat import agent_router as router  # noqa: F401
