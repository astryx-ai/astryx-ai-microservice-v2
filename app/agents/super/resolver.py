from __future__ import annotations
from typing import Any, Dict, List, Optional
import re

from app.tools.azure_openai import chat_model
from langchain.prompts import ChatPromptTemplate

from .state import AgentState, TickerRecord
from .utils import (
    parse_intent,
    parse_news_detail,
    classify_intent,
    yahoo_autocomplete,
)


def _normalize_company_query(name: str) -> str:
    n = (name or "").strip().casefold()
    n = re.sub(r"[\"'`]+", "", n)
    suffixes = r"\b(ltd\.?|limited|pvt\.?|private|inc\.?|co\.?|company|corp\.?|corporation)\b"
    n = re.sub(suffixes, "", n, flags=re.I)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def _levenshtein_ratio(a: str, b: str) -> float:
    """Compute a simple Levenshtein-based similarity ratio in [0,1]."""
    a, b = a or "", b or ""
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0.0
    dp = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        dp[i][0] = i
    for j in range(lb + 1):
        dp[0][j] = j
    for i in range(1, la + 1):
        ca = a[i - 1]
        for j in range(1, lb + 1):
            cb = b[j - 1]
            cost = 0 if ca == cb else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    dist = dp[la][lb]
    return max(0.0, 1.0 - dist / max(la, lb))


def _get_ticker_record_from_memory(memory: Dict[str, Any]) -> Optional[TickerRecord]:
    """Load TickerRecord from memory if present; fallback to legacy keys."""
    try:
        m = (memory or {}).get("ticker_record")
        if isinstance(m, dict):
            cn = m.get("company_name") or ""
            ns = m.get("nse_symbol")
            bs = m.get("bse_symbol")
            if cn or ns or bs:
                return TickerRecord(company_name=cn, nse_symbol=ns, bse_symbol=bs)
    except Exception:
        pass
    try:
        comp = (memory or {}).get("company")
        tkr = (memory or {}).get("ticker")
        exch = (memory or {}).get("exchange")
        if comp and tkr:
            if exch == "NSE":
                return TickerRecord(company_name=comp, nse_symbol=tkr, bse_symbol=None)
            if exch == "BSE":
                return TickerRecord(company_name=comp, nse_symbol=None, bse_symbol=tkr)
            return TickerRecord(company_name=comp, nse_symbol=tkr, bse_symbol=None)
    except Exception:
        pass
    return None



def _llm_extract_company(question: str) -> Optional[str]:
    """Use Azure OpenAI to extract the single, official Indian (NSE/BSE) company name."""
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are a precise extractor of Indian listed company names for NSE/BSE.
Task: Given a user query, output the single official company name as registered on NSE/BSE.
Rules:
- Expand tickers/acronyms to full names (e.g., RIL -> Reliance Industries Limited, TCS -> Tata Consultancy Services Limited, LTTS -> L&T Technology Services Limited, HDFC -> HDFC Bank Limited).
- Normalize informal names to their official counterparts (e.g., Reliance -> Reliance Industries Limited, Jio -> Reliance Jio Infocomm Limited, Adani enterprises -> Adani Enterprises Limited).
- Focus on the most likely company mentioned; if ambiguous or no clear company, return {"company": null}.
- Use exact full names as they appear in official listings, including 'Limited' if applicable.
- Output JSON only, no extra text.
""".strip(),
        ),
        (
            "human",
            """
User query: {q}
Respond with JSON exactly as:
{"company": "<Official Indian company name>"} or {"company": null}
Examples:
Input: "stock of RIL" -> {"company": "Reliance Industries Limited"}
Input: "price of tcs" -> {"company": "Tata Consultancy Services Limited"}
Input: "news of adani enterprises" -> {"company": "Adani Enterprises Limited"}
Input: "l and t technology price" -> {"company": "L&T Technology Services Limited"}
Input: "what about reliance" -> {"company": "Reliance Industries Limited"}
Input: "jindal drilling" -> {"company": "Jindal Drilling and Industries Limited"}
""".strip(),
        ),
    ])
    try:
        res = (prompt | chat_model(temperature=0.0)).invoke({"q": question})
        content = getattr(res, "content", "") or ""
        import json, re as _re
        m = _re.search(r"\{.*\}", content, _re.S)
        if not m:
            return None
        data = json.loads(m.group(0))
        comp = (data or {}).get("company")
        if comp and isinstance(comp, str) and comp.strip():
            return comp.strip()
        return None
    except Exception:
        return None




def _extract_company_phrase(q: str) -> str:
    s = (q or "").strip()
    m = re.search(r"[\"']([^\"']{2,80})[\"']", s)
    if m:
        return m.group(1).strip()
    m = re.search(r"\b(?:for|of|on|about|what about)\s+([A-Za-z0-9&\./\-\s]{2,80})", s, re.I)
    if m:
        phrase = re.sub(r"\s+(?:on|in|with|using|chart|line|area|bar|candle|candlestick|ohlc)\b.*$", "", m.group(1), flags=re.I)
        return phrase.strip()
    # Strip common intent/aux words; if nothing meaningful remains, return empty
    cleaned = re.sub(r"\b(price|stock|quote|chart|today|news|headline|update|latest|share|company|details|info|about)\b", " ", s, flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # If only punctuation or nothing left, no explicit company phrase
    if not cleaned or not re.search(r"[A-Za-z0-9&]", cleaned):
        return ""
    return cleaned


def resolve_ticker_node(state: AgentState) -> AgentState:
    question = state.get("question", "")
    memory = state.get("memory", {})

    # First, set intent/detail using robust classifier
    state["intent"] = classify_intent(question, memory)
    state["news_detail"] = parse_news_detail(question)

    # Skip only pure greetings; for clarify/others we still try to resolve
    if state["intent"] == "greeting":
        return state

    # 1) Extract likely company phrase; if ambiguous text (e.g., "Ambani's company"), use LLM once.
    phrase = _extract_company_phrase(question)
    norm = _normalize_company_query(phrase)
    # Quick alias map for common acronyms/colloquialisms; avoids LLM when unavailable
    alias_map = {
        "ril": "Reliance Industries Limited",
        "reliance": "Reliance Industries Limited",
        "ambani": "Reliance Industries Limited",
        "ambani's": "Reliance Industries Limited",
        "ambanis": "Reliance Industries Limited",
        "tcs": "Tata Consultancy Services Limited",
        "infy": "Infosys Limited",
        "lt": "Larsen & Toubro Limited",
        "l&t": "Larsen & Toubro Limited",
        "hdfc": "HDFC Bank Limited",
    }
    if norm in alias_map:
        phrase, norm = alias_map[norm], _normalize_company_query(alias_map[norm])
    elif not norm or any(k in norm for k in ["ambani", "ambani's", "ambani s"]):
        comp = _llm_extract_company(question)
        if comp:
            phrase, norm = comp, _normalize_company_query(comp)

    # Memory short-circuit: if user didn't specify a new company, reuse last TickerRecord
    mem_rec = _get_ticker_record_from_memory(memory)
    if mem_rec:
        mem_norm = _normalize_company_query(mem_rec.company_name or "")
        same_company = bool(mem_norm and norm and mem_norm == norm)
        no_new_company = not norm
        if (no_new_company or same_company) and not re.search(r"\b(new|different|change|switch)\b", question or "", re.I):
            state["company"] = mem_rec.company_name
            if mem_rec.nse_symbol:
                state["ticker"] = mem_rec.nse_symbol
                state["exchange"] = "NSE"
                state["full_symbol"] = f"{mem_rec.nse_symbol}.NS"
            elif mem_rec.bse_symbol:
                state["ticker"] = mem_rec.bse_symbol
                state["exchange"] = "BSE"
                state["full_symbol"] = f"{mem_rec.bse_symbol}.BO"
            if state.get("intent") not in ("stock", "news", "both"):
                state["intent"] = "both"
            return state

    # 2) Call Yahoo autocomplete with the phrase.
    cands = yahoo_autocomplete(phrase or question, limit=8)
    if not cands:
        state["intent"] = "clarify"
        return state

    # 3) Rank: prefer NSE/BSE; exact/prefix/levenshtein; Yahoo score as tie-breaker.
    best = None
    best_score = -1.0
    single_root = len(norm.split()) == 1
    for c in cands:
        cname = (c.get("company") or "").strip()
        compn = _normalize_company_query(cname)
        lev = _levenshtein_ratio(norm, compn) if norm else 0.0
        phrase_bonus = 0.8 if norm and norm in compn else 0.0
        prefix_bonus = 0.8 if compn.startswith(norm) else 0.0
        ex_bonus = 0.5 if c.get("exchange") == "NSE" else (0.3 if c.get("exchange") == "BSE" else 0.0)
        yscore = float(c.get("score") or 0.0) / 10000.0
        # Prefer parent entities for single-root queries
        parent_bonus = 0.0
        if single_root:
            if "industries" in compn:
                parent_bonus += 0.6
            if any(x in compn for x in ["power", "green", "ports", "capital", "transmission", "energy"]):
                parent_bonus -= 0.2
        s = lev * 5.0 + phrase_bonus + prefix_bonus + ex_bonus + yscore + parent_bonus
        if s > best_score:
            best, best_score = c, s

    # 4) Decide: high confidence → set state; else show suggestions and clarify.
    confident = best is not None and (best_score >= 2.8 or len(cands) == 1)
    if confident and best:
        symbol = best.get("symbol") or ""
        longname = best.get("longname") or best.get("company") or ""
        exch = best.get("exchange")
        # Populate state for downstream stock/news nodes
        state["company"] = longname
        # strip .NS/.BO for our internal ticker
        if symbol.upper().endswith(".NS"):
            state["ticker"] = symbol[:-3]
            state["exchange"] = "NSE"
        elif symbol.upper().endswith(".BO"):
            state["ticker"] = symbol[:-3]
            state["exchange"] = "BSE"
        else:
            state["ticker"] = best.get("ticker")
            state["exchange"] = exch
        # also keep full symbol metadata if needed elsewhere
        state["full_symbol"] = symbol
        state["sector"] = best.get("sector")
        state["industry"] = best.get("industry")
    else:
        # Build suggestion list of 4–5 items
        query_snip = phrase or question
        sugg = []
        for c in cands[:5]:
            sugg.append({
                "company": c.get("longname") or c.get("company"),
                "symbol": c.get("symbol"),
                "ticker": c.get("ticker"),
                "exchange": c.get("exchange"),
                "sector": c.get("sector"),
                "industry": c.get("industry"),
            })
        state["suggestions"] = sugg
        state["intent"] = "clarify"
        state["query_snip"] = query_snip
        return state

    # Persist memory using TickerRecord plus legacy keys for compatibility
    try:
        comp = state.get("company") or ""
        tkr = state.get("ticker")
        ex = state.get("exchange")
        nse_symbol: Optional[str] = None
        bse_symbol: Optional[str] = None
        if ex == "NSE":
            nse_symbol = tkr
        elif ex == "BSE":
            bse_symbol = tkr
        if tkr and not nse_symbol and not bse_symbol:
            nse_symbol = tkr
        memory["ticker_record"] = {
            "company_name": comp,
            "nse_symbol": nse_symbol,
            "bse_symbol": bse_symbol,
        }
        memory.update({
            "company": comp,
            "ticker": tkr,
            "exchange": ex,
        })
    except Exception:
        memory.update({
            "company": state.get("company"),
            "ticker": state.get("ticker"),
            "exchange": state.get("exchange"),
        })
    state["memory"] = memory
    # If intent wasn't stock/news/both but we resolved a ticker, default to both
    if state.get("intent") not in ("stock", "news", "both"):
        state["intent"] = "both"
    return state
