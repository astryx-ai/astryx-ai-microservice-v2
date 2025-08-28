from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from datetime import datetime, timedelta, timezone

from .state import AgentState
from .utils import fmt_money, change_emoji, brand_from_url, strip_urls


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

    if any(v for v in [high, low, mcap, vol] if v and v != "-"):
        table = (
            f"**Stock Snapshot** for **{company} ({ticker} - {ex})**\n\n"
            "|---\n"
            "| **Metric**      | **Value**       |\n"
            "|-----------------|-----------------|\n"
            f"| Price           | {price} {emoji} {pct_str} |\n"
            f"| High / Low      | {high} / {low}  |\n"
            f"| Market Cap      | {mcap}          |\n"
            f"| Volume          | {vol}           |\n"
            "|---\n\n"
        )
    else:
        table = (
            f"**Stock Snapshot** for **{company} ({ticker} - {ex})**\n\n"
            f"- **Price**: {price} {emoji} {pct_str}\n\n"
            "---\n"
        )
    return table


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

    # Greeting/clarify shortcuts
    if intent == "greeting":
        state["output"] = "Hello! Ask me about any NSE/BSE stock or the latest company news."
        return state
    if intent == "clarify":
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

    output = (
        f"Here’s the scoop on {company}{f' ({ticker})' if ticker else ''} as of "
        f"{datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime('%I:%M %p IST, %B %d, %Y')}:\n\n"
    )

    if intent in ("stock", "both") and stock:
        output += format_stock_section(company, ticker, ex, stock) + "\n"

    if intent in ("news", "both") and news:
        output += news_section(news, detail)

    if not (stock or news):
        output = (
            "Oops! No data found for the Indian market. Try a specific NSE/BSE ticker like "
            "'TATAMOTORS' or 'NIFTY'!\n"
        )

    state["output"] = output
    return state
