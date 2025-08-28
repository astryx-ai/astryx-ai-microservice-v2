"""LangGraph scaffolding for incremental migration.

This package contains:
- state: typed state schemas shared across graphs
- checkpoints: checkpoint saver factory (sqlite for now)
- utils: helpers (timing, retries)
- pipelines: per-pipeline graphs (news, chat, charts, etc.)
- wrappers: feature-flag dispatchers for routes/gRPC

All wrappers are drop-in and preserve existing outputs.
"""
