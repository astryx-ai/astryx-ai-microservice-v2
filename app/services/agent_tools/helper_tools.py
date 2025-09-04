from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable


def get_current_datetime_string() -> str:
    """Return current UTC date/time in a readable, unambiguous format."""
    now = datetime.now(timezone.utc)
    # Example: 2025-09-04 12:34 UTC
    return now.strftime("%Y-%m-%d %H:%M UTC")


def _contains_any(text: str, needles: Iterable[str]) -> bool:
    lowered = (text or "").lower()
    return any(n in lowered for n in needles)


def needs_recency_injection(query: str) -> bool:
    """Heuristic to detect when the user likely wants the latest/most current info.

    Triggers on phrases like: today, now, current, latest, recent, this week, past week,
    as of, up to date, breaking, new, just released, update, updated, real-time.
    """
    keywords = [
        "today",
        "now",
        "current",
        "latest",
        "recent",
        "this week",
        "past week",
        "as of",
        "up to date",
        "uptodate",
        "breaking",
        "new",
        "just released",
        "update",
        "updated",
        "real-time",
        "realtime",
        "live",
        "this month",
        "this quarter",
        "this year",
        "ytd",
    ]
    decision = _contains_any(query, keywords)
    if decision:
        print(f"[Helper] needs_recency_injection: True for query='{query}'")
    else:
        print(f"[Helper] needs_recency_injection: False for query='{query}'")
    return decision


def inject_datetime_into_query(query: str) -> str:
    """Append an "as of <DATE TIME>" suffix to the query for recency-sensitive searches."""
    injected = f"{query} (as of {get_current_datetime_string()})".strip()
    print(f"[Helper] inject_datetime_into_query: '{injected}'")
    return injected


def requires_deep_research(query: str, has_context: bool = False) -> bool:
    """Determine if a query requires deep research based on keywords and complexity."""
    print(f"[Helper] Evaluating if query requires deep research: '{query}' (has_context: {has_context})")
    
    query_lower = query.lower()
    
    # Quick follow-up questions that should use standard search
    simple_follow_up_patterns = [
        "quick comparison", "quick summary", "what we discussed", "what did we", 
        "summarize", "summary", "briefly", "in short", "you mentioned",
        "companies you mentioned", "what about", "how about", "tell me about"
    ]
    
    for pattern in simple_follow_up_patterns:
        if pattern in query_lower:
            print(f"[Helper] Standard search sufficient - found simple follow-up pattern: '{pattern}'")
            return False
    
    # If query references previous context and is short, likely a follow-up
    if has_context and len(query.split()) <= 10:
        context_references = ["you mentioned", "companies", "those", "these", "them", "it", "that"]
        if any(ref in query_lower for ref in context_references):
            print("[Helper] Standard search sufficient - short query with context reference")
            return False
    
    # Keywords that indicate comprehensive analysis is needed (more selective)
    deep_research_indicators = [
        "deep research", "comprehensive analysis", "detailed analysis",
        "market analysis", "industry overview", "strategic analysis",
        "competitive landscape", "business model analysis"
    ]
    
    # Check for explicit deep research requests
    for indicator in deep_research_indicators:
        if indicator in query_lower:
            print(f"[Helper] Deep research required - found explicit indicator: '{indicator}'")
            return True
    
    # Complex multi-topic queries (more restrictive)
    if ("and" in query_lower and len(query.split()) > 12) or len(query.split()) > 15:
        print("[Helper] Deep research required - very complex query")
        return True
    
    # Check for initial research requests (without context)
    if not has_context:
        initial_research_patterns = [
            "top", "best", "leading", "major", "key players", "market leaders",
            "future outlook", "growth prospects", "market trends"
        ]
        word_count = len(query.split())
        if word_count > 8 and any(pattern in query_lower for pattern in initial_research_patterns):
            print("[Helper] Deep research required - initial comprehensive research request")
            return True
    
    print("[Helper] Standard search sufficient for this query")
    return False

