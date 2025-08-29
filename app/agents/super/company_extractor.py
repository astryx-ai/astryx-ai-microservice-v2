from __future__ import annotations

"""
Company extractor pipeline with stages:
1) Regex + cleanup (preserves existing rules)
2) NER (spaCy or HuggingFace, optional)
3) Fuzzy DB match (rapidfuzz if available, else difflib)
4) LLM fallback using Azure OpenAI (extract_entities_tool)

Returns a JSON-serializable list of objects with fields:
- name (canonical from DB)
- nse (NSE symbol if any)
- bse (BSE symbol if any)
- confidence (0-100)

Also includes a SAMPLE_DB and demo queries when run as a script.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import json
import logging
import re
import difflib

try:  # Optional fast fuzzy
    from rapidfuzz import fuzz as rf_fuzz
    from rapidfuzz import process as rf_process
except Exception:  # pragma: no cover
    rf_fuzz = None  # type: ignore
    rf_process = None  # type: ignore

try:  # Optional spaCy NER
    import spacy  # type: ignore
except Exception:  # pragma: no cover
    spacy = None  # type: ignore

try:  # Optional HF transformers NER
    from transformers import pipeline as hf_pipeline  # type: ignore
except Exception:  # pragma: no cover
    hf_pipeline = None  # type: ignore

from app.tools.azure_openai import chat_model
from langchain.prompts import ChatPromptTemplate


# --------------------------- Data structures ---------------------------


@dataclass
class CompanyRecord:
    name: str
    nse: Optional[str] = None
    bse: Optional[str] = None


# --------------------------- Helpers & Config ---------------------------


OWNER_TO_COMPANY = {
    # owner/brand -> canonical company
    "mukesh ambani": "Reliance Industries Limited",
    "ambani": "Reliance Industries Limited",
}

ABBREVIATIONS = {
    # common tickers/short names -> canonical
    "RIL": "Reliance Industries Limited",
    "RELIANCE": "Reliance Industries Limited",
    "TCS": "Tata Consultancy Services Limited",
    "INFY": "Infosys Limited",
    "LT": "Larsen & Toubro Limited",
    "L&T": "Larsen & Toubro Limited",
    "LTTS": "L&T Technology Services Limited",
}


def _norm(text: str) -> str:
    s = (text or "").strip()
    s = re.sub(r"[\"'`]+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _norm_cmp(text: str) -> str:
    s = _norm(text).casefold()
    # remove common suffixes
    s = re.sub(r"\b(ltd\.?|limited|pvt\.?|private|inc\.?|co\.?|company|corp\.?|corporation)\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# --------------------------- Stage 1: Regex ---------------------------


def _split_multi_phrases(phrase: str) -> List[str]:
    # Split on commas and and/or
    parts = re.split(r"\s*(?:,|\band\b|\bor\b)\s*", phrase, flags=re.I)
    return [p for p in (part.strip() for part in parts) if p]


def _regex_extract(query: str) -> List[str]:
    """Preserve existing regex cleanup rules and return candidate phrases (can be multiple)."""
    s = (query or "").strip()
    out: List[str] = []

    # quoted phrase
    m = re.search(r"[\"']([^\"']{2,80})[\"']", s)
    if m:
        out.extend(_split_multi_phrases(m.group(1).strip()))

    # for|of|on|about|what about group with trailing cleanup
    m = re.search(r"\b(?:for|of|on|about|what about)\s+([A-Za-z0-9&\./\-\s]{2,80})", s, re.I)
    if m:
        phrase = re.sub(r"\s+(?:on|in|with|using|chart|line|area|bar|candle|candlestick|ohlc)\b.*$", "", m.group(1), flags=re.I)
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
        out.extend(_split_multi_phrases(phrase.strip()))

    # general cleanup of query to find freeform candidates
    cleaned = re.sub(
        r"\b(price|stock|quote|chart|today|news|headline|update|latest|share|company|details|info|about|both|and|or|please|show|give|tell|me)\b",
        " ",
        s,
        flags=re.I,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned and re.search(r"[A-Za-z0-9&]", cleaned):
        out.extend(_split_multi_phrases(cleaned))

    # Dedup while preserving order
    seen = set()
    uniq: List[str] = []
    for p in out:
        k = _norm_cmp(p)
        if k and k not in seen and len(k) > 1:
            seen.add(k)
            uniq.append(p)
    return uniq


# --------------------------- Stage 2: NER ---------------------------


_NER = None  # cached spaCy or HF pipeline


def _load_ner():  # lazy
    global _NER
    if _NER is not None:
        return _NER
    # Try spaCy first
    if spacy is not None:
        try:  # prefer already installed models
            nlp = spacy.load("en_core_web_sm")
            _NER = ("spacy", nlp)
            return _NER
        except Exception:
            pass
    # Try HF transformers
    if hf_pipeline is not None:
        try:
            ner = hf_pipeline("ner", aggregation_strategy="simple")
            _NER = ("hf", ner)
            return _NER
        except Exception:
            pass
    _NER = ("none", None)
    return _NER


def _ner_extract(query: str) -> List[str]:
    mode, ner = _load_ner()
    ents: List[str] = []
    if mode == "spacy" and ner is not None:
        try:
            doc = ner(query)
            ents = [e.text for e in doc.ents if e.label_ in {"ORG", "PRODUCT"}]
        except Exception:
            ents = []
    elif mode == "hf" and ner is not None:
        try:
            preds = ner(query)
            ents = [p["word"] for p in preds if (p.get("entity_group") or p.get("entity")).startswith("ORG")]
        except Exception:
            ents = []
    # Light heuristic fallback: capitalized chunks with company-like words
    if not ents:
        m = re.findall(r"\b([A-Z][A-Za-z&\-\.]+(?:\s+[A-Z][A-Za-z&\-\.]+){0,3})\b", query)
        ents = m or []
    # Dedup and return
    seen = set()
    out: List[str] = []
    for e in ents:
        k = _norm_cmp(e)
        if k and k not in seen:
            seen.add(k)
            out.append(_norm(e))
    return out


# --------------------------- Stage 3: Fuzzy DB Match ---------------------------


def _score_ratio(a: str, b: str) -> int:
    if rf_fuzz:
        try:
            # Use a robust combo to handle typos and token differences
            scores = [
                rf_fuzz.WRatio(a, b),
                rf_fuzz.token_set_ratio(a, b),
                rf_fuzz.partial_ratio(a, b),
            ]
            return int(max(scores))
        except Exception:
            return int(rf_fuzz.token_set_ratio(a, b))
    # difflib fallback ~ [0..100]
    return int(round(100 * difflib.SequenceMatcher(None, a, b).ratio()))


def _best_match(name: str, db: Sequence[CompanyRecord], threshold: int = 70) -> Optional[Tuple[CompanyRecord, int]]:
    if not name:
        return None
    ncmp = _norm_cmp(name)
    if not ncmp:
        return None
    # Abbreviation normalization
    upper_tok = _norm(name).upper()
    if upper_tok in ABBREVIATIONS:
        canon = ABBREVIATIONS[upper_tok]
        ncmp = _norm_cmp(canon)
    # owner mapping
    lower = _norm(name).casefold()
    if lower in OWNER_TO_COMPANY:
        canon = OWNER_TO_COMPANY[lower]
        ncmp = _norm_cmp(canon)

    best: Optional[Tuple[CompanyRecord, int]] = None
    for rec in db:
        sc = _score_ratio(ncmp, _norm_cmp(rec.name))
        if sc >= threshold:
            if best is None or sc > best[1]:
                best = (rec, sc)
    return best


def _fuzzy_match(name: str, db: Sequence[Dict[str, Any] | CompanyRecord]) -> Optional[Tuple[CompanyRecord, int]]:
    # Accept dict or CompanyRecord
    db_rec: List[CompanyRecord] = []
    for r in db:
        if isinstance(r, CompanyRecord):
            db_rec.append(r)
        else:
            db_rec.append(CompanyRecord(name=r.get("name") or r.get("company_name") or "", nse=r.get("nse") or r.get("nse_symbol"), bse=r.get("bse") or r.get("bse_symbol")))
    return _best_match(name, db_rec, threshold=70)


# --------------------------- Stage 4: LLM Fallback ---------------------------


def extract_entities_tool(query: str) -> List[str]:
    """Use Azure OpenAI to extract company names from a query.

    Returns a list of strings. JSON-only contract in the prompt.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert extractor of publicly listed company names in India (NSE/BSE).\n"
            "Given a user query, extract any company names mentioned. Expand tickers/abbreviations\n"
            "(e.g., RIL->Reliance Industries Limited, TCS->Tata Consultancy Services Limited).\n"
            "If the query refers to an owner or brand (e.g., Mukesh Ambani company), infer the likely listed parent.\n"
            "Respond as JSON only, no extra text."
        )),
        ("human", (
            "Query: {q}\n"
            "Return JSON exactly as {\"companies\": [\"<name1>\", \"<name2>\", ...]}\n"
            "Examples:\n"
            "'show stock of TCS and Infosys' -> {\"companies\": [\"Tata Consultancy Services Limited\", \"Infosys Limited\"]}\n"
            "'Mukesh Ambani company news' -> {\"companies\": [\"Reliance Industries Limited\"]}\n"
        )),
    ])
    try:
        res = (prompt | chat_model(temperature=0.0)).invoke({"q": query})
        content = getattr(res, "content", "") or ""
        m = re.search(r"\{.*\}", content, re.S)
        if not m:
            return []
        data = json.loads(m.group(0))
        arr = data.get("companies") or []
        return [str(x).strip() for x in arr if isinstance(x, str) and str(x).strip()]
    except Exception:
        return []


def _llm_fallback(query: str, db: Sequence[Dict[str, Any] | CompanyRecord]) -> List[Dict[str, Any]]:
    names = extract_entities_tool(query)
    results: List[Dict[str, Any]] = []
    for n in names:
        m = _fuzzy_match(n, db)
        if not m:
            continue
        rec, sc = m
        confidence = max(60, min(95, sc))  # slightly conservative for LLM-derived
        results.append({
            "name": rec.name,
            "nse": rec.nse,
            "bse": rec.bse,
            "confidence": int(confidence),
        })
    return _dedup_results(results)


# --------------------------- Orchestrator ---------------------------


def _dedup_results(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for it in items:
        key = _norm_cmp(it.get("name") or "")
        if key and key not in seen:
            seen.add(key)
            out.append(it)
        else:
            # if duplicate with higher confidence exists, keep highest
            for o in out:
                if _norm_cmp(o.get("name") or "") == key:
                    o["confidence"] = max(int(o.get("confidence") or 0), int(it.get("confidence") or 0))
                    break
    return out


def extract_company(query: str, db: Sequence[Dict[str, Any] | CompanyRecord]) -> List[Dict[str, Any]]:
    """Full pipeline orchestrator. Returns list of {name,nse,bse,confidence}."""
    candidates: List[str] = []

    # Stage 1: Regex
    candidates.extend(_regex_extract(query))

    # Stage 2: NER
    candidates.extend(_ner_extract(query))

    # Canonicalize owner/abbr expansions into candidates as well
    extra: List[str] = []
    for c in list(candidates) or []:
        up = _norm(c).upper()
        if up in ABBREVIATIONS:
            extra.append(ABBREVIATIONS[up])
        low = _norm(c).casefold()
        if low in OWNER_TO_COMPANY:
            extra.append(OWNER_TO_COMPANY[low])
    candidates.extend(extra)

    # Dedup candidates by normalized compare key
    seen = set()
    uniq: List[str] = []
    for c in candidates:
        k = _norm_cmp(c)
        if k and k not in seen:
            seen.add(k)
            uniq.append(c)

    # Stage 3: Fuzzy match
    results: List[Dict[str, Any]] = []
    for c in uniq:
        m = _fuzzy_match(c, db)
        if not m:
            continue
        rec, sc = m
        # Confidence mapping: fuzzy score, slight boost if candidate came from abbr/owner
        boost = 5 if (c.upper() in ABBREVIATIONS or c.casefold() in OWNER_TO_COMPANY) else 0
        confidence = max(50, min(100, sc + boost))
        results.append({
            "name": rec.name,
            "nse": rec.nse,
            "bse": rec.bse,
            "confidence": int(confidence),
        })

    results = _dedup_results(results)

    if results:
        return results

    # Stage 4: LLM fallback
    return _llm_fallback(query, db)


# --------------------------- Sample DB & Demo ---------------------------


SAMPLE_DB: List[CompanyRecord] = [
    CompanyRecord("Reliance Industries Limited", nse="RELIANCE", bse="500325"),
    CompanyRecord("Tata Consultancy Services Limited", nse="TCS", bse="532540"),
    CompanyRecord("Infosys Limited", nse="INFY", bse="500209"),
    CompanyRecord("Larsen & Toubro Limited", nse="LT", bse="500510"),
    CompanyRecord("L&T Technology Services Limited", nse="LTTS", bse="540115"),
    CompanyRecord("Adani Enterprises Limited", nse="ADANIENT", bse="512599"),
]


DEMO_QUERIES = [
    "candlestick chart for Relaince",
    "show me stock of TCS and Infosys",
    "Mukesh Ambani company news",
    "price for RIL",
]


def _demo_run():
    print("Demo: company extractor pipeline\n")
    for q in DEMO_QUERIES:
        res = extract_company(q, SAMPLE_DB)
        print(f"Query: {q}")
        print(json.dumps(res, ensure_ascii=False))
        print()


if __name__ == "__main__":  # pragma: no cover
    _demo_run()
