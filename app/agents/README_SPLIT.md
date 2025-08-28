Super Agent split

- Canonical public API lives in app/agents/super/runner.py
- Graph composition is in app/agents/super/graph.py
- Nodes are split into resolver.py, stock.py, news.py, and formatting.py
- Keep using from app.agents.super.runner import run_super_agent
- The legacy shim app/services/super_agent.py can be removed once all deployments are updated.
