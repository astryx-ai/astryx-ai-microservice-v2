"""Deprecated shim module for /super/*.

The actual shim routes are defined in app.routes.chat (super_router).
We re-export its router here to preserve imports.
"""

from app.routes.chat import super_router as router  # noqa: F401
