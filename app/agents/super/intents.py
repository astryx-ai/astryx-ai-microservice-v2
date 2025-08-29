from __future__ import annotations
from typing import Dict, Any, List, Optional, Set, Tuple
import re


_RE_STOCK = re.compile(r"\b(price|stock|quote|ohlc|pe|market\s*cap|volume|high|low|intraday|today)\b", re.I)
_RE_NEWS = re.compile(r"\b(news|headline|article|report|update|latest)\b", re.I)
_RE_CHART = re.compile(r"\b(chart|candle|candlestick|ohlc|line\s*chart|area\s*chart|bar\s*chart)\b", re.I)
_RE_EXPAND = re.compile(r"\b(elaborate|more\s*details|expand|tell\s*me\s*more|deepen|summary|summarize)\b", re.I)
_RE_GREET = re.compile(r"\b(hi|hello|hey|how\s*are\s*you|good\s*(morning|afternoon|evening)|what's\s*up)\b", re.I)


def classify_multi_intent(query: str, memory: Optional[Dict[str, Any]] = None) -> List[str]:
    """Return a list of intents among: stock, news, chart, casual, expand_news.

    LLM-first: let the model choose multiple tools; fall back to minimal regex if LLM unavailable.
    """
    q = (query or "").strip()
    if not q:
        return ["casual"]

    allowed = {"stock", "news", "chart", "casual", "expand_news"}
    # 1) LLM router (primary)
    try:
        routed = llm_tool_router(q, memory)
        tools = routed.get("tools") or []
        tools = [str(t).lower() for t in tools if str(t).lower() in allowed]
        if tools:
            s = set(tools)
            # If the user explicitly asked for stock only, drop news/chart even if LLM added them
            explicit_stock = bool(_RE_STOCK.search(q))
            explicit_news = bool(_RE_NEWS.search(q))
            explicit_chart = bool(_RE_CHART.search(q))
            # If the user explicitly asked for chart only, do not add stock/news implicitly
            if explicit_chart and not (explicit_stock or explicit_news):
                return ["chart"]
            if explicit_stock and not (explicit_news or explicit_chart):
                return ["stock"]
            # Ensure chart is present if explicitly requested but LLM missed it
            if explicit_chart and "chart" not in s:
                s.add("chart")
            return list(s)
    except Exception:
        pass

    # 2) Minimal regex fallback only when LLM fails/returns nothing
    intents: Set[str] = set()
    if _RE_GREET.search(q):
        intents.add("casual")
    has_chart = bool(_RE_CHART.search(q))
    has_stock = bool(_RE_STOCK.search(q))
    has_news = bool(_RE_NEWS.search(q))
    if has_chart:
        intents.add("chart")
    if has_stock:
        intents.add("stock")
    if has_news:
        intents.add("news")
    if _RE_EXPAND.search(q):
        intents.add("expand_news")

    if not intents:
        has_ctx = bool(memory and (memory.get("ticker") or memory.get("company")))
        if has_ctx:
            intents.update(["stock", "news"])
        else:
            intents.add("casual")
    # Stricter rule: if user asked only for stock, don't add news/chart implicitly
    if has_stock and not (has_news or has_chart):
        return ["stock"]
    # Stricter rule: if user asked only for chart, don't add stock/news implicitly
    if has_chart and not (has_stock or has_news):
        return ["chart"]
    return list(intents)


def wants_expand_only(intents: List[str]) -> bool:
    s = set(intents or [])
    return ("expand_news" in s) and (s <= {"expand_news", "casual"})


def llm_tool_router(query: str, memory: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Use the LLM to parse the user's request and recommend tools.

    Returns a dict like: {
      "tools": ["stock","news","chart"|"casual"|"expand_news"],
      "entities": {"companies": ["Reliance"], "timeframe": "1d 5m"}
    }
    """
    try:
        from langchain.prompts import ChatPromptTemplate  # type: ignore
        from app.tools.azure_openai import chat_model  # type: ignore
    except Exception:
        # No LLM available; return empty
        return {"tools": [], "entities": {}}

    mem_snip = ""
    if memory:
        try:
            parts = []
            for k in ("company","ticker","exchange","last_news_link"):
                v = memory.get(k)
                if v:
                    parts.append(f"{k}={v}")
            mem_snip = ", ".join(parts)
        except Exception:
            mem_snip = ""

    schema = (
        '{"tools": ["stock"|"news"|"chart"|"casual"|"expand_news"...], '
        '"entities": {"companies": [string...], "timeframe": string|null}}'
    )
    prompt = ChatPromptTemplate.from_template(
        """You are a router for a financial assistant with tools: stock, news, chart, casual, expand_news.
Decide which tools to call for the message and extract any companies and timeframe hints.
Rules:
- Combine tools when the user asks multiple things (e.g., "news and chart of TCS").
- If user asks to elaborate more on last news/article, include expand_news.
- If the user requests only a chart, choose only chart. Do not add stock unless they also ask for price/metrics.
- If small talk, choose only casual.
Return JSON exactly with keys tools and entities. No extra text.

Context: {memory}
Message: {query}
Output JSON schema: {schema}
"""
    )
    try:
        resp = (prompt | chat_model(temperature=0.0)).invoke({
            "memory": mem_snip,
            "query": query,
            "schema": schema,
        })
        raw = getattr(resp, "content", "") or ""
        import json
        import re as _re
        m = _re.search(r"\{.*\}", raw, _re.S)
        data = json.loads(m.group(0)) if m else {}
        tools = data.get("tools") or []
        if not isinstance(tools, list):
            tools = []
        # Normalize
        tools = [t for t in [str(x).lower() for x in tools] if t in {"stock","news","chart","casual","expand_news"}]
        entities = data.get("entities") or {}
        return {"tools": tools, "entities": entities}
    except Exception:
        return {"tools": [], "entities": {}}
