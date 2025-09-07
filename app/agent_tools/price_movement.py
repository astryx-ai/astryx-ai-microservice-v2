"""
Price Movement Analysis Tool
Fetches 1-month stock data from Upstox and correlates with news using OpenAI.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from langchain.tools import tool
import requests
import logging
import json

from app.config import settings
from app.utils.model import CandleQuery, CandleSeries
from app.services.upstox import UpstoxProvider
from app.agent_tools.exa import exa_search
from app.services.llms.azure_openai import chat_model
from app.utils.stream_utils import emit_process
from app.db import companies as companies_repo

logger = logging.getLogger(__name__)


class PriceMovementInput(BaseModel):
    """Input schema for price movement analysis."""
    company_query: str = Field(description="Company name or ticker symbol (e.g., 'RELIANCE', 'TCS', 'Infosys')")
    instrument_key: Optional[str] = Field(default=None, description="Upstox instrument key (optional, will be resolved from company name)")


def _resolve_instrument_key(company_query: str) -> Optional[str]:
    """
    Resolve company name to Upstox instrument key using the companies search,
    mirroring the stock_screen route behavior. Builds key as BSE_EQ%7C{ISIN}.
    """
    q = (company_query or "").strip()
    if not q:
        return None

    # Exact/ILIKE search
    matches = companies_repo.search_companies(q, limit=1)
    isin: Optional[str] = None
    if matches:
        isin = matches[0].get("isin")

    # Fuzzy fallback
    if not isin:
        fuzzy = companies_repo.fuzzy_search_companies(q, limit=1)
        if fuzzy:
            isin = fuzzy[0].get("isin")

    if not isin:
        return None

    return f"BSE_EQ%7C{isin}"


def _get_monthly_candle_data(instrument_key: str) -> Optional[CandleSeries]:
    """
    Fetch ~5 months candle data from Upstox (monthly interval).
    """
    try:
        # Calculate date range (~last 5 months)
        to_date = datetime.now()
        from_date = to_date - timedelta(days=150)
        
        # Format dates for Upstox API (YYYY-MM-DD)
        to_date_str = to_date.strftime("%Y-%m-%d")
        from_date_str = from_date.strftime("%Y-%m-%d")
        
        query = CandleQuery(
            instrument_key=instrument_key,
            unit="months",
            interval="1",
            from_date=from_date_str,
            to_date=to_date_str
        )
        
        upstox = UpstoxProvider()
        print(f"[Tool] fetching candle data for {query} from {from_date_str} to {to_date_str}")
        raw_data = upstox.fetch_historical_candles(query)
        
        # Convert to canonical format
        candle_series = UpstoxProvider.convert_upstox_candles_to_canonical(raw_data, query)
        
        return candle_series
        
    except Exception as e:
        logger.error(f"Error fetching candle data for {instrument_key}: {str(e)}")
        return None


def _calculate_monthly_metrics(candle_series: CandleSeries, n: int = 5) -> List[Dict[str, Any]]:
    """Return last n monthly metrics: [{month, open, close, high, low, return_pct}] most recent first."""
    metrics: List[Dict[str, Any]] = []
    try:
        candles = sorted(candle_series.candles, key=lambda c: c.timestamp)
        for c in candles[-n:]:
            try:
                month = str(c.timestamp)[:7]
            except Exception:
                month = str(c.timestamp)
            try:
                ret = ((c.close - c.open) / c.open) * 100 if c.open else None
            except Exception:
                ret = None
            metrics.append({
                "month": month,
                "open": c.open,
                "close": c.close,
                "high": c.high,
                "low": c.low,
                "return_pct": (round(ret, 2) if ret is not None else None),
            })
        metrics = list(reversed(metrics))  # most recent first
    except Exception:
        metrics = []
    return metrics


def _get_company_news(company_query: str) -> str:
    """
    Fetch recent news articles about the company using Exa.
    """
    try:
        # Create a focused news search query for the last 6 months and include month tokens
        months_tokens = []
        try:
            today = datetime.utcnow().date().replace(day=1)
            for i in range(6):
                m = (today.replace(day=1) - timedelta(days=1)).replace(day=1) if i > 0 else today
                months_tokens.append(m.strftime("%Y-%m"))
                today = m
        except Exception:
            months_tokens = []
        month_hint = " ".join(months_tokens)
        news_query = f"{company_query} stock price news earnings financial results last 6 months {month_hint}"
        
        # Use exa_search to get news articles
        news_results = exa_search.func(news_query, max_results=15)
        print(f"[Tool] news_results: {news_results}")
        
        return news_results
        
    except Exception as e:
        logger.error(f"Error fetching news for {company_query}: {str(e)}")
        return f"Unable to fetch news for {company_query}: {str(e)}"


def _analyze_price_news_correlation(monthly_metrics: List[Dict[str, Any]],  news_content: str, company_name: str) -> str:
    """
    Use OpenAI to analyze correlation between monthly closes and news.
    """
    try:
        print(f"[Tool] analyzing price-news correlation for {company_name} {news_content}")
        llm = chat_model(temperature=0.3)
        months_json = json.dumps(monthly_metrics or [])
        
        analysis_prompt = f"""
You are a senior financial analyst. Produce a concise month-on-month view for up to the last 5 months and correlate each month with notable news.

MONTHLY_DATA (JSON):
{months_json}

NEWS (verbatim excerpts):
{news_content}

INSTRUCTIONS (strict):
- Use MONTHLY_DATA to anchor the last 5 months by calendar month (YYYY-MM). If fewer than 5 rows are present, output fewer.
- Correlate each month with the single most plausible reason from NEWS. If no relevant news, set reason to "N/A".
- Return ONLY raw JSON (no code fences, no prose) as an array of objects with this exact shape: [{{"month":"YYYY-MM","reason":"..."}}]
"""
        print(f"[Tool] analysis_prompt: {analysis_prompt}")
        response = llm.invoke(analysis_prompt)
        return response.content
        
    except Exception as e:
        logger.error(f"Error in AI analysis: {str(e)}")
        return f"Error analyzing correlation: {str(e)}"


@tool("price_movement_analysis")
def price_movement_analysis(company_query: str, instrument_key: Optional[str] = None) -> str:
    """
    Analyze stock price movements over the last months and correlate with news events.
    """
    print(f"[Tool] price_movement_analysis called | company='{company_query}', instrument_key='{instrument_key}'")
    emit_process({"message": f"Analyzing price movements for {company_query}"})
    
    try:
        # Step 1: Resolve instrument key if not provided
        if not instrument_key:
            instrument_key = _resolve_instrument_key(company_query)
            if not instrument_key:
                return f"❌ Could not resolve instrument key for '{company_query}'. Please provide a valid company name or ticker symbol."
        
        emit_process({"message": f"Fetching 5-month stock data for {company_query}"})
        
        # Step 2: Get monthly candle data from Upstox
        candle_data = _get_monthly_candle_data(instrument_key)
        if not candle_data or not candle_data.candles:
            return f"❌ Could not fetch stock data for {company_query} (instrument: {instrument_key}). Please check if the company ticker is valid."
        
        # Step 3: Build last-5-months metrics array
        monthly_metrics = _calculate_monthly_metrics(candle_data, n=5)
        print(f"[Tool] monthly_metrics: {monthly_metrics}")
        if not monthly_metrics:
            return f"❌ Insufficient monthly data to analyze for {company_query}."
        
        emit_process({"message": f"Fetching recent news for {company_query}"})
        
        # Step 4: Get company news
        news_content = _get_company_news(company_query)
        
        emit_process({"message": f"Analyzing price-news correlation for {company_query}"})
        
        # Step 5: AI-powered correlation analysis
        correlation_analysis = _analyze_price_news_correlation(monthly_metrics, news_content, company_query)
        
        return correlation_analysis
        
    except Exception as e:
        logger.exception(f"Error in price movement analysis for {company_query}")
        return f"❌ Error during price movement analysis for {company_query}: {str(e)}"


# Create structured tool for use in subgraphs
from langchain.tools import StructuredTool

price_movement_tool = StructuredTool.from_function(
    func=price_movement_analysis,
    name="price_movement_analysis",
    description="Analyze stock price movements over the last months and correlate with news events using Upstox data and Exa news.",
    args_schema=PriceMovementInput,
)
