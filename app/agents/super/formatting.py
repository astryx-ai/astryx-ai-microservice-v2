from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from datetime import datetime, timedelta, timezone

from .state import AgentState
from .company_extractor import _regex_extract as _regex_extract_candidates, extract_company as extract_company_pipeline
from .resolver import _load_company_db_subset
from app.interfaces.grpc.client import call_stock_via_grpc_or_local, call_news_via_grpc_or_local
from .utils import fmt_money, change_emoji, brand_from_url, strip_urls
from .intents import classify_multi_intent
from .casual import casual_response


def format_stock_section(company: str, ticker: str, ex: str, stock: Dict[str, Any]) -> str:
    if not stock:
        return ""
    price = fmt_money(stock.get("current_price"))
    pct = stock.get("percent_change")
    pct_str = f"{pct:.2f}%" if pct is not None else "-"
    high = fmt_money(stock.get("daily_high"))
    low = fmt_money(stock.get("daily_low"))
    mcap = fmt_money(stock.get("market_cap"))
    vol = fmt_money(stock.get("volume"))
    emoji = change_emoji(pct)
    rows = [
        ("Price", f"{price} {emoji} {pct_str}"),
    ]
    if high != "-" or low != "-":
        rows.append(("High / Low", f"{high} / {low}"))
    if mcap != "-":
        rows.append(("Market Cap", mcap))
    if vol != "-":
        rows.append(("Volume", vol))
    # If only price is available, use a minimal bullet style
    if len(rows) == 1:
        return (
            f"**Stock Snapshot** for **{company} ({ticker} - {ex})**\n\n"
            f"- **Price**: {price} {emoji} {pct_str}\n\n---\n"
        )
    # Else, render a compact table
    table = [f"**Stock Snapshot** for **{company} ({ticker} - {ex})**\n\n", "|---\n", "| **Metric**      | **Value**       |\n", "|-----------------|-----------------|\n"]
    for k, v in rows:
        table.append(f"| {k:<15} | {v} |\n")
    table.append("|---\n\n")
    return "".join(table)


def _limit_sentences(text: str, max_sentences: int, max_words: int) -> str:
    text = strip_urls(text)
    import re

    sentences = re.split(r"(?<=[.!?])\s+", text)
    out: List[str] = []
    wc = 0
    for s in sentences[:max_sentences]:
        words = s.split()
        if wc + len(words) > max_words:
            break
        out.append(" ".join(words))
        wc += len(words)
    summary = " ".join(out)
    if summary and not re.search(r"[.!?]$", summary):
        summary += "."
    return summary


def news_section(items: List[Dict[str, Any]], detail: Literal["short", "medium", "long"]) -> str:
    if not items:
        return "\n\n*No recent news available. Try again later!*"
    if detail == "short":
        max_sent, max_words = 1, 30
    elif detail == "long":
        max_sent, max_words = 5, 150
    else:
        max_sent, max_words = 3, 80
    bullets = []
    for it in items[:4]:
        title = it.get("title", "Untitled")
        summary = _limit_sentences(it.get("summary", ""), max_sent, max_words)
        url = it.get("url", "")
        if url:
            brand = brand_from_url(url)
            tag = f" [{brand}]({url})"
        else:
            tag = ""
        bullets.append(f"- **{title}**: {summary}{tag}")
    return f"\n\n**Recent News** ✨:\n" + "\n".join(bullets) + "\n---"


def merge_results_node(state: AgentState) -> AgentState:
    company = state.get("company") or "Market"
    ticker = state.get("ticker") or ""
    ex = state.get("exchange") or ""
    intent = state.get("intent", "both")
    stock = state.get("stock_data") or {}
    news = state.get("news_items") or []
    detail = state.get("news_detail", "medium")

    # If chart intent present, acknowledge chart generation (chart JSON is handled via gRPC internally)
    intents = state.get("intents", [])  # type: ignore
    if intents and "chart" in intents:
        # Resolve a friendly symbol string
        extras = state.get("extras", {}) or {}
        chart_meta_sym = None
        try:
            chart_meta_sym = ((extras.get("chart") or {}).get("meta") or {}).get("symbol")
        except Exception:
            chart_meta_sym = None
        # Try to fallback to memory if state lacks symbol
        mem = state.get("memory", {}) or {}
        sym = chart_meta_sym or state.get("full_symbol") or (
            f"{(state.get('ticker') or '').upper()}.NS" if state.get("exchange") == "NSE" else (
                f"{(state.get('ticker') or '').upper()}.BO" if state.get("exchange") == "BSE" else (state.get('ticker') or '').upper()
            )
        )
        if not sym:
            mt, me = (mem.get("ticker") or ""), (mem.get("exchange") or "")
            if mt and me == "NSE":
                sym = f"{mt.upper()}.NS"
            elif mt and me == "BSE":
                sym = f"{mt.upper()}.BO"
            else:
                sym = mt.upper() or "(symbol)"
        chart_ack = f"Your chart for {sym} is ready. Want price or latest news as well?"
    # If only chart (no stock/news), return early with the acknowledgement
        if not stock and not news and intent not in ("stock", "news", "both"):
            # If we reused memory, prepend an acknowledgement
            try:
                if (state.get("extras", {}) or {}).get("resolved_via_memory"):
                    ack_name = (state.get("company") or "").strip() or ((state.get("memory", {}) or {}).get("last_company") or "the last company")
                    chart_ack = f"Showing you the latest for {ack_name} since that was your last company.\n\n" + chart_ack
            except Exception:
                pass
            # Persist compact last_* memory fields even on early return
            try:
                mem = state.get("memory", {}) or {}
                comp = state.get("company")
                tkr = state.get("ticker")
                exx = state.get("exchange")
                if comp and comp != "Market":
                    mem["last_company"] = comp
                if tkr:
                    if exx == "NSE":
                        mem["last_ticker"] = f"{tkr.upper()}.NS"
                    elif exx == "BSE":
                        mem["last_ticker"] = f"{tkr.upper()}.BO"
                    else:
                        mem["last_ticker"] = tkr.upper()
                mem["last_intent"] = "chart" if "chart" in (intents or []) else (state.get("intent") or "")
                state["memory"] = mem
            except Exception:
                pass
            state["output"] = chart_ack
            return state

    # Greeting/clarify shortcuts
    if intent == "greeting":
        state["output"] = "Hello! Ask me about any NSE/BSE stock or the latest company news."
        return state
    if intent == "clarify" and not (intents and "chart" in intents):
        sugg = state.get("suggestions") or []
        if sugg:
            bullets = []
            for s in sugg:
                comp = s.get("company") or "Unknown"
                sym = s.get("symbol") or ""
                sec = s.get("sector") or ""
                ind = s.get("industry") or ""
                meta = " - ".join([x for x in [sec, ind] if x])
                line = f"- {comp} ({sym}){f' — {meta}' if meta else ''}"
                bullets.append(line)
            q = state.get("query_snip") or "this query"
            msg = (
                f"I found multiple matches for \"{q}\":\n" +
                "\n".join(f"{i+1}. {b[2:]}" for i, b in enumerate(bullets)) +
                "\nWhich one do you want?"
            )
            state["output"] = msg
        else:
            state["output"] = "Do you want stock details, news, or both? You can say 'Price of TCS' or 'Latest news on HDFC Bank'."
        return state

    if intents and set(intents) <= {"casual"}:
        state["output"] = casual_response(state.get("question") or "")
        return state

    ts = datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime('%I:%M %p IST, %B %d, %Y')
    # Persist compact last_* memory fields for vague follow-ups
    try:
        mem = state.get("memory", {}) or {}
        # last_company
        if company and company != "Market":
            mem["last_company"] = company
        # last_ticker in Yahoo symbol style if exchange available
        sym = None
        if ticker:
            if ex == "NSE":
                sym = f"{ticker.upper()}.NS"
            elif ex == "BSE":
                sym = f"{ticker.upper()}.BO"
            else:
                sym = ticker.upper()
            mem["last_ticker"] = sym
        # last_intent based on intents/intent
        intents_list = state.get("intents", [])  # type: ignore
        if intents_list:
            if "chart" in intents_list and ("stock" in intents_list or intent in ("stock","both")) and ("news" in intents_list or intent in ("news","both")):
                mem["last_intent"] = "stock+news+chart"
            elif "chart" in intents_list and ("stock" in intents_list or intent in ("stock","both")):
                mem["last_intent"] = "stock+chart"
            elif "chart" in intents_list:
                mem["last_intent"] = "chart"
            elif "stock" in intents_list and "news" in intents_list:
                mem["last_intent"] = "stock+news"
            elif "stock" in intents_list:
                mem["last_intent"] = "stock"
            elif "news" in intents_list:
                mem["last_intent"] = "news"
        else:
            mem["last_intent"] = intent
        state["memory"] = mem
    except Exception:
        pass
    multi = state.get("multi_results") or []
    # Fallback: if user likely asked multiple companies but we only fetched one, try to expand here
    q = state.get("question", "") or ""
    if not multi:
        try:
            cands = _regex_extract_candidates(q)
        except Exception:
            cands = []
        if len(cands) >= 2:
            subset = _load_company_db_subset(q)
            uniq = {}
            for c in cands[:5]:
                sub = extract_company_pipeline(c, subset) or []
                if not sub:
                    continue
                top = max(sub, key=lambda x: int(x.get("confidence") or 0))
                key = (top.get("name") or "").strip().lower()
                if key and key not in uniq:
                    uniq[key] = top
            matches = list(uniq.values())
            # If we already have a primary ticker, keep it first and add others
            prim_key = (company or "").strip().lower()
            ordered = []
            if prim_key:
                for m in matches:
                    if (m.get("name") or "").strip().lower() == prim_key:
                        ordered.append(m)
                for m in matches:
                    if m not in ordered:
                        ordered.append(m)
            else:
                ordered = matches
            # Fetch stock/news for top 2-3 entries
            built = []
            for m in ordered[:5]:
                m_t = m.get("nse") or m.get("bse")
                m_e = "NSE" if m.get("nse") else ("BSE" if m.get("bse") else None)
                m_c = m.get("name")
                sd = call_stock_via_grpc_or_local(ticker=m_t, exchange=m_e, company=m_c) if ("stock" in intents or "both" in intents) else None
                # Fetch news even when chart is requested to enrich the response
                nd = call_news_via_grpc_or_local(ticker=m_t, company=m_c) if (("news" in intents) or ("chart" in intents) or ("both" in intents)) else None
                built.append({
                    "company": m_c,
                    "ticker": m_t,
                    "exchange": m_e,
                    "stock_data": sd,
                    "news_items": nd or [],
                })
            if built:
                multi = built
                state["multi_results"] = multi  # type: ignore
    output = ""
    # If we reused context from memory, add a friendly acknowledgment line
    try:
        if (state.get("extras", {}) or {}).get("resolved_via_memory"):
            ack_name = company if company and company != "Market" else (state.get("memory", {}) or {}).get("last_company") or "the last company"
            output += f"Showing you the latest for {ack_name} since that was your last company.\n\n"
    except Exception:
        pass
    if multi:
        # At-a-glance summary table for stocks (if requested)
        header = f"Here’s the scoop on {len(multi)} companies as of {ts}:\n\n"
        sections = [header]
        if intent in ("stock", "both"):
            lines = ["| Company (Ticker - EX) | Price | Change |", "|---|---:|---:|"]
            for m in multi:
                m_company = m.get("company") or "Company"
                m_ticker = (m.get("ticker") or "").upper()
                m_ex = m.get("exchange") or ""
                sd = m.get("stock_data") or {}
                price = fmt_money(sd.get("current_price")) if sd else "-"
                pct = sd.get("percent_change") if sd else None
                emoji = change_emoji(pct)
                pct_str = f"{pct:.2f}%" if pct is not None else "-"
                lines.append(f"| {m_company} ({m_ticker} - {m_ex}) | {price} | {emoji} {pct_str} |")
            sections.append("\n" + "\n".join(lines) + "\n\n")
        # Per-company detail blocks (stock/news)
        for m in multi:
            m_company = m.get("company") or "Company"
            m_ticker = m.get("ticker") or ""
            m_ex = m.get("exchange") or ""
            m_stock = m.get("stock_data") or {}
            m_news = m.get("news_items") or []
            if intent in ("stock", "both") and m_stock:
                sections.append(format_stock_section(m_company, m_ticker, m_ex, m_stock))
            # Show news if requested explicitly, or when chart is requested (we fetched news alongside charts)
            if (intent in ("news", "both") or ("chart" in intents)) and m_news:
                sections.append(news_section(m_news, detail))
            sections.append("\n")
        output = "".join(sections).strip() + "\n"
    else:
        output = (
            f"Here’s the scoop on {company}{f' ({ticker})' if ticker else ''} as of {ts}:\n\n"
        )
        if intent in ("stock", "both") and stock:
            output += format_stock_section(company, ticker, ex, stock) + "\n"
        # Show news if requested explicitly, or when chart is requested (we fetched news alongside charts)
        if (intent in ("news", "both") or ("chart" in intents)) and news:
            output += news_section(news, detail)

    # If neither single-company stock/news exists, check multi-company results before failing
    if not (stock or news):
        has_multi_content = False
        try:
            for m in (state.get("multi_results") or []):
                if (m.get("stock_data") or m.get("news_items")):
                    has_multi_content = True
                    break
        except Exception:
            has_multi_content = False
        if has_multi_content:
            state["output"] = output
            return state
        intents_list = state.get("intents", [])  # type: ignore
        if intents_list and "chart" in intents_list:
            # Preserve the earlier chart acknowledgement; optionally add a hint
            if not output.strip():
                sym = state.get("full_symbol") or (f"{(state.get('ticker') or '').upper()}.NS" if state.get("exchange") == "NSE" else (f"{(state.get('ticker') or '').upper()}.BO" if state.get("exchange") == "BSE" else (state.get('ticker') or 'the symbol')))
                output = f"Your chart for {sym} is generated and sent via the backend. Want price or latest news as well?"
        else:
            output = (
                "Oops! No data found for the Indian market. Try a specific NSE/BSE ticker like "
                "'TATAMOTORS' or 'NIFTY'!\n"
            )

    state["output"] = output
    return state
