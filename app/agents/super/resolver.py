from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
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
from .company_extractor import extract_company as extract_company_pipeline, SAMPLE_DB
from .company_extractor import _regex_extract as _regex_extract_candidates

# Optional Supabase client for company DB lookups
try:  # pragma: no cover
    from supabase import create_client  # type: ignore
    from app.tools.config import settings  # type: ignore
except Exception:  # pragma: no cover
    create_client = None  # type: ignore
    settings = None  # type: ignore


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


def _supabase_client():
    """Create and cache a Supabase client if settings are present."""
    global _SB
    try:
        _SB
    except NameError:  # first time
        _SB = None  # type: ignore
    if _SB is not None:
        return _SB
    if not create_client or not settings:
        _SB = None  # type: ignore
        return _SB
    try:
        _SB = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)  # type: ignore
    except Exception:
        _SB = None  # type: ignore
    return _SB


def _fuzzy_match_company(query: str, rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Choose the best matching company row using fuzzy name similarity.

    rows elements should have: company_name, nse_symbol, bse_symbol
    """
    norm_q = _normalize_company_query(query)
    single_root = len(norm_q.split()) == 1
    best: Optional[Dict[str, Any]] = None
    best_score = -1.0
    for r in rows or []:
        cname = (r.get("company_name") or "").strip()
        compn = _normalize_company_query(cname)
        lev = _levenshtein_ratio(norm_q, compn) if norm_q else 0.0
        contains = 0.8 if norm_q and norm_q in compn else 0.0
        prefix = 0.8 if compn.startswith(norm_q) else 0.0
        parent = 0.0
        if single_root:
            if "industries" in compn:
                parent += 0.6
            if any(x in compn for x in ["power", "green", "ports", "capital", "transmission", "energy"]):
                parent -= 0.2
        s = lev * 5.0 + contains + prefix + parent
        # Prefer NSE presence slightly
        if r.get("nse_symbol"):
            s += 0.3
        if s > best_score:
            best, best_score = r, s
    return best


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
    # New: consider compact last_* fields if present
    try:
        last_comp = (memory or {}).get("last_company")
        last_tkr = (memory or {}).get("last_ticker")  # may be RELIANCE.NS style or raw ticker
        if last_comp or last_tkr:
            ns = None
            bs = None
            if isinstance(last_tkr, str) and last_tkr:
                up = last_tkr.upper().strip()
                if up.endswith(".NS"):
                    ns = up[:-3]
                elif up.endswith(".BO"):
                    bs = up[:-3]
                else:
                    # assume NSE as default if no suffix
                    ns = up
            return TickerRecord(company_name=last_comp or "", nse_symbol=ns, bse_symbol=bs)
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



def ticker_tool(raw_input: str) -> Optional[TickerRecord]:
    """Resolve raw company/ticker input to a TickerRecord using Companies DB with fuzzy match.

    Returns TickerRecord with both NSE and BSE symbols where available. Falls back to None.
    """
    s = (raw_input or "").strip()
    if not s:
        return None
    # If explicit Yahoo-style symbol
    up = s.upper()
    if up.endswith(".NS"):
        return TickerRecord(company_name="", nse_symbol=up[:-3], bse_symbol=None)
    if up.endswith(".BO"):
        return TickerRecord(company_name="", nse_symbol=None, bse_symbol=up[:-3])
    # Try exact symbol match first for short uppercase tokens
    sb = _supabase_client()
    try:
        norm = _normalize_company_query(s)
        upper_q = norm.upper()
        is_short_upper = len(upper_q) <= 6 and upper_q == s.upper()
        row = None
        if sb and is_short_upper:
            res = sb.table("companies").select("company_name,nse_symbol,bse_symbol").or_(
                f"nse_symbol.eq.{upper_q},bse_symbol.eq.{upper_q}"
            ).limit(1).execute()
            if res and res.data:
                row = res.data[0]
        if not row and sb:
            toks = [t for t in re.split(r"\W+", norm) if t]
            if toks:
                clauses = [f"company_name.ilike.%{t}%" for t in toks[:3]]
            else:
                clauses = [f"company_name.ilike.%{norm}%"]
            res = sb.table("companies").select("company_name,nse_symbol,bse_symbol").or_(",".join(clauses)).limit(100).execute()
            rows = res.data or []
            row = _fuzzy_match_company(s, rows)
        if row:
            return TickerRecord(company_name=row.get("company_name") or "", nse_symbol=row.get("nse_symbol"), bse_symbol=row.get("bse_symbol"))
    except Exception:
        pass
    # Fallback to Yahoo autocomplete top hit
    try:
        cands = yahoo_autocomplete(s, limit=5)
        for c in cands:
            sym = (c.get("symbol") or "").upper()
            if not sym:
                continue
            longname = c.get("longname") or c.get("company") or ""
            if sym.endswith(".NS"):
                return TickerRecord(company_name=longname, nse_symbol=sym[:-3], bse_symbol=None)
            if sym.endswith(".BO"):
                return TickerRecord(company_name=longname, nse_symbol=None, bse_symbol=sym[:-3])
    except Exception:
        pass
    return None


def _extract_company_phrase(q: str) -> str:
    s = (q or "").strip()
    m = re.search(r"[\"']([^\"']{2,80})[\"']", s)
    if m:
        return m.group(1).strip()
    m = re.search(r"\b(?:for|of|on|about|what about)\s+([A-Za-z0-9&\./\-\s]{2,80})", s, re.I)
    if m:
        phrase = re.sub(r"\s+(?:on|in|with|using|chart|line|area|bar|candle|candlestick|ohlc)\b.*$", "", m.group(1), flags=re.I)
        # Extra cleanup: stop at common pronouns/fillers after the company keyword
        STOP_AFTER = {"i", "me", "need", "all", "them", "please", "thanks", "thank", "you", "now", "today"}
        toks = [t for t in re.split(r"\W+", phrase) if t]
        if toks:
            kept: List[str] = []
            for t in toks:
                if t.lower() in STOP_AFTER:
                    break
                kept.append(t)
            if kept:
                phrase = " ".join(kept)
        return phrase.strip()
    # Strip common intent/aux words; if nothing meaningful remains, return empty
    cleaned = re.sub(r"\b(price|stock|quote|chart|today|news|headline|update|latest|share|company|details|info|about|both|and|or|please|show|give|tell|me)\b", " ", s, flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # If only punctuation, stopwords, or too short, no explicit company phrase
    STOP = {"and", "or", "the", "a", "an", "me", "more", "please"}
    if not cleaned or not re.search(r"[A-Za-z0-9&]", cleaned):
        return ""
    toks = re.split(r"\W+", cleaned)
    toks = [t for t in toks if t]
    if not toks:
        return ""
    if len(toks) == 1 and (len(toks[0]) <= 2 or toks[0].lower() in STOP):
        return ""
    # Extremely short fragments like 'and', 'both', 'more' should not count as company phrases
    if cleaned.lower() in STOP or len(cleaned) <= 2:
        return ""
    return cleaned


def extract_company(query: str, db: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Adapter around the new company extractor pipeline.

    If no DB is provided, uses a small SAMPLE_DB for demonstration.
    """
    _db = db if db is not None else [
        {"name": r.name, "nse": r.nse, "bse": r.bse} for r in SAMPLE_DB
    ]
    return extract_company_pipeline(query, _db)


def _load_company_db_subset(query: str) -> List[Dict[str, Any]]:
    """Fetch a small subset of companies from DB relevant to the query tokens.

    Falls back to SAMPLE_DB if Supabase isn't configured.
    """
    tokens = [t for t in re.split(r"\W+", _normalize_company_query(query)) if t]
    STOP = {"and","or","for","of","on","about","what","chart","charts","candle","candlestick","ohlc","price","stock","quote","today","news","headline","update","latest","share","company","details","info","both","please","show","give","tell","me"}
    tokens = [t for t in tokens if len(t) >= 2 and t not in STOP][:6]
    sb = _supabase_client()
    rows: List[Dict[str, Any]] = []
    if sb and tokens:
        try:
            clauses = [f"company_name.ilike.%{t}%" for t in tokens]
            res = sb.table("companies").select("company_name,nse_symbol,bse_symbol").or_(",".join(clauses)).limit(300).execute()
            for r in (res.data or []):
                rows.append({
                    "name": r.get("company_name"),
                    "nse": r.get("nse_symbol"),
                    "bse": r.get("bse_symbol"),
                })
        except Exception:
            rows = []
    # Always include a small seed set to help resolve typos/common cases
    seed = [{"name": r.name, "nse": r.nse, "bse": r.bse} for r in SAMPLE_DB]
    rows.extend(seed)
    # Dedup by normalized name
    seen = set()
    dedup: List[Dict[str, Any]] = []
    for r in rows:
        key = _normalize_company_query(r.get("name") or "").casefold()
        if key and key not in seen:
            seen.add(key)
            dedup.append(r)
    rows = dedup
    return rows


def resolve_ticker_node(state: AgentState) -> AgentState:
    question = state.get("question", "")
    memory = state.get("memory", {})

    # First, set intent/detail using robust classifier
    state["intent"] = classify_intent(question, memory)
    state["news_detail"] = parse_news_detail(question)

    # Skip only pure greetings; for clarify/others we still try to resolve
    if state["intent"] == "greeting":
        return state

    # Memory short-circuit: if no explicit company extracted, consider reuse later.
    mem_rec = _get_ticker_record_from_memory(memory)

    # 1) Use the new company extractor pipeline with a DB subset
    db_subset = _load_company_db_subset(question)
    matches = extract_company_pipeline(question, db_subset)
    # If user likely requested multiple companies, resolve each candidate independently
    try:
        cand_list = _regex_extract_candidates(question)
    except Exception:
        cand_list = []
    agg: List[Dict[str, Any]] = []
    if len(cand_list) >= 2:
        seen = set()
        for c in cand_list[:5]:
            sub = extract_company_pipeline(c, db_subset)
            if not sub:
                continue
            # pick top per candidate
            top = max(sub, key=lambda x: int(x.get("confidence") or 0))
            key = (_normalize_company_query(top.get("name") or "")).casefold()
            if key and key not in seen:
                seen.add(key)
                agg.append(top)
        if agg:
            matches = agg

    if matches:
        # Keep all matches when multiple companies requested
        state["matches"] = matches
        # Choose the top for immediate ticker resolution
        top = max(matches, key=lambda x: int(x.get("confidence") or 0))
        state["company"] = top.get("name")
        nse = top.get("nse")
        bse = top.get("bse")
        if nse:
            state["ticker"] = nse
            state["exchange"] = "NSE"
            state["full_symbol"] = f"{nse}.NS"
        elif bse:
            state["ticker"] = bse
            state["exchange"] = "BSE"
            state["full_symbol"] = f"{bse}.BO"
        # Persist memory
        memory["ticker_record"] = {
            "company_name": state.get("company"),
            "nse_symbol": nse,
            "bse_symbol": bse,
        }
        memory.setdefault("company", state.get("company"))
        if state.get("ticker"):
            memory.setdefault("ticker", state["ticker"])
            memory.setdefault("exchange", state.get("exchange"))
        state["memory"] = memory
        if state.get("intent") not in ("stock", "news", "both"):
            state["intent"] = "both"
        return state

    # 2) If pipeline couldn't find any, reuse memory if intent demands
    if state.get("intent") in ("stock", "news", "both") and mem_rec is not None and not re.search(r"\b(new|different|change|switch)\b", question or "", re.I):
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
        # Mark that we resolved context via memory to enable user-facing acknowledgement
        try:
            state.setdefault("extras", {}).update({"resolved_via_memory": True})
        except Exception:
            pass
        return state

    # 3) DB failed — Call Yahoo autocomplete with the phrase and rank.
    phrase = _extract_company_phrase(question)
    norm = _normalize_company_query(phrase) if phrase else _normalize_company_query(question)
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
