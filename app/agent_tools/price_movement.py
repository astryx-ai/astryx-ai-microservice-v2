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

from app.config import settings
from app.utils.model import CandleQuery, CandleSeries
from app.services.upstox import UpstoxProvider
from app.agent_tools.exa import exa_search
from app.services.llms.azure_openai import chat_model
from app.utils.stream_utils import emit_process

logger = logging.getLogger(__name__)


class PriceMovementInput(BaseModel):
    """Input schema for price movement analysis."""
    company_query: str = Field(description="Company name or ticker symbol (e.g., 'RELIANCE', 'TCS', 'Infosys')")
    instrument_key: Optional[str] = Field(default=None, description="Upstox instrument key (optional, will be resolved from company name)")


def _resolve_instrument_key(company_query: str) -> Optional[str]:
    """
    Resolve company name to Upstox instrument key.
    For demo purposes, using common mappings.
    In production, this would query a comprehensive database.
    """
    # Common Indian stock mappings for demo
    mappings = {
        "reliance": "NSE_EQ|INE002A01018",
        "tcs": "NSE_EQ|INE467B01029", 
        "infosys": "NSE_EQ|INE009A01021",
        "hdfc bank": "NSE_EQ|INE040A01034",
        "icici bank": "NSE_EQ|INE090A01021",
        "bharti airtel": "NSE_EQ|INE397D01024",
        "itc": "NSE_EQ|INE154A01025",
        "sbi": "NSE_EQ|INE062A01020",
        "wipro": "NSE_EQ|INE075A01022",
        "maruti suzuki": "NSE_EQ|INE585B01010",
        "bajaj finance": "NSE_EQ|INE296A01024",
        "hul": "NSE_EQ|INE030A01027",
        "adani enterprises": "NSE_EQ|INE423A01024",
        "coal india": "NSE_EQ|INE522F01014",
        "ntpc": "NSE_EQ|INE733E01010"
    }
    
    query_lower = company_query.lower().strip()
    
    # Direct match
    if query_lower in mappings:
        return mappings[query_lower]
    
    # Partial match
    for key, value in mappings.items():
        if query_lower in key or key in query_lower:
            return value
    
    # If no match found, try to construct NSE_EQ format if it looks like a ticker
    if len(company_query) <= 10 and company_query.isalpha():
        logger.warning(f"No mapping found for {company_query}, using generic NSE_EQ format")
        return f"NSE_EQ|{company_query.upper()}"
    
    return None


def _get_monthly_candle_data(instrument_key: str) -> Optional[CandleSeries]:
    """
    Fetch 1-month candle data from Upstox.
    """
    try:
        # Calculate date range (last 30 days)
        to_date = datetime.now()
        from_date = to_date - timedelta(days=30)
        
        # Format dates for Upstox API (YYYY-MM-DD)
        to_date_str = to_date.strftime("%Y-%m-%d")
        from_date_str = from_date.strftime("%Y-%m-%d")
        
        query = CandleQuery(
            instrument_key=instrument_key,
            unit="days",
            interval="1",
            from_date=from_date_str,
            to_date=to_date_str
        )
        
        upstox = UpstoxProvider()
        raw_data = upstox.fetch_historical_candles(query)
        
        # Convert to canonical format
        candle_series = UpstoxProvider.convert_upstox_candles_to_canonical(raw_data, query)
        
        return candle_series
        
    except Exception as e:
        logger.error(f"Error fetching candle data for {instrument_key}: {str(e)}")
        return None


def _calculate_price_metrics(candle_series: CandleSeries) -> Dict[str, Any]:
    """
    Calculate key price movement metrics from candle data.
    """
    if not candle_series.candles:
        return {}
    
    candles = candle_series.candles
    
    # Sort by timestamp to ensure chronological order
    sorted_candles = sorted(candles, key=lambda x: x.timestamp)
    
    if len(sorted_candles) < 2:
        return {}
    
    # Calculate metrics
    first_candle = sorted_candles[0]
    last_candle = sorted_candles[-1]
    
    # Price movement
    price_change = last_candle.close - first_candle.open
    price_change_pct = (price_change / first_candle.open) * 100
    
    # High and Low for the month
    monthly_high = max(candle.high for candle in sorted_candles)
    monthly_low = min(candle.low for candle in sorted_candles)
    
    # Volatility (simple range-based)
    volatility = ((monthly_high - monthly_low) / first_candle.open) * 100
    
    # Volume analysis (if available)
    total_volume = sum(candle.volume or 0 for candle in sorted_candles)
    avg_volume = total_volume / len(sorted_candles) if total_volume > 0 else 0
    
    return {
        "period": f"{first_candle.timestamp} to {last_candle.timestamp}",
        "opening_price": first_candle.open,
        "closing_price": last_candle.close,
        "monthly_high": monthly_high,
        "monthly_low": monthly_low,
        "price_change": round(price_change, 2),
        "price_change_percentage": round(price_change_pct, 2),
        "volatility_percentage": round(volatility, 2),
        "total_volume": total_volume,
        "average_daily_volume": round(avg_volume, 0),
        "trading_days": len(sorted_candles)
    }


def _get_company_news(company_query: str) -> str:
    """
    Fetch recent news articles about the company using Exa.
    """
    try:
        # Create a focused news search query
        news_query = f"{company_query} stock price news earnings financial results last month"
        
        # Use exa_search to get news articles
        news_results = exa_search.func(news_query, max_results=8)
        
        return news_results
        
    except Exception as e:
        logger.error(f"Error fetching news for {company_query}: {str(e)}")
        return f"Unable to fetch news for {company_query}: {str(e)}"


def _analyze_price_news_correlation(price_metrics: Dict[str, Any], news_content: str, company_name: str) -> str:
    """
    Use OpenAI to analyze correlation between price movements and news.
    """
    try:
        llm = chat_model(temperature=0.3)
        
        analysis_prompt = f"""
You are a financial analyst expert. Analyze the correlation between stock price movements and news events for {company_name}.

PRICE MOVEMENT DATA (Last 30 days):
- Period: {price_metrics.get('period', 'N/A')}
- Opening Price: ₹{price_metrics.get('opening_price', 'N/A')}
- Closing Price: ₹{price_metrics.get('closing_price', 'N/A')}
- Monthly High: ₹{price_metrics.get('monthly_high', 'N/A')}
- Monthly Low: ₹{price_metrics.get('monthly_low', 'N/A')}
- Price Change: ₹{price_metrics.get('price_change', 'N/A')} ({price_metrics.get('price_change_percentage', 'N/A')}%)
- Volatility: {price_metrics.get('volatility_percentage', 'N/A')}%
- Average Daily Volume: {price_metrics.get('average_daily_volume', 'N/A'):,}
- Trading Days: {price_metrics.get('trading_days', 'N/A')}

NEWS & EVENTS:
{news_content}

ANALYSIS REQUIREMENTS:
1. **Price Movement Summary**: Summarize the key price movements and trends
2. **News Impact Analysis**: Identify specific news events that likely influenced price movements
3. **Correlation Insights**: Explain how news events correlate with price changes
4. **Key Drivers**: Highlight the most significant factors affecting the stock
5. **Market Sentiment**: Assess overall market sentiment based on news and price action

Provide a comprehensive analysis in a structured format with clear insights and actionable information.
"""

        response = llm.invoke(analysis_prompt)
        return response.content
        
    except Exception as e:
        logger.error(f"Error in AI analysis: {str(e)}")
        return f"Error analyzing correlation: {str(e)}"


@tool("price_movement_analysis")
def price_movement_analysis(company_query: str, instrument_key: Optional[str] = None) -> str:
    """
    Analyze stock price movements over the last month and correlate with news events.
    
    This tool fetches 1-month candle data from Upstox, gets recent news from Exa,
    and uses AI to analyze correlations between price movements and news events.
    
    Args:
        company_query: Company name or ticker symbol (e.g., 'RELIANCE', 'TCS', 'Infosys')
        instrument_key: Optional Upstox instrument key (will be auto-resolved if not provided)
    
    Returns:
        Comprehensive analysis of price movements correlated with news events
    """
    print(f"[Tool] price_movement_analysis called | company='{company_query}', instrument_key='{instrument_key}'")
    emit_process({"message": f"Analyzing price movements for {company_query}"})
    
    try:
        # Step 1: Resolve instrument key if not provided
        if not instrument_key:
            instrument_key = _resolve_instrument_key(company_query)
            if not instrument_key:
                return f"❌ Could not resolve instrument key for '{company_query}'. Please provide a valid company name or ticker symbol."
        
        emit_process({"message": f"Fetching 1-month stock data for {company_query}"})
        
        # Step 2: Get monthly candle data from Upstox
        candle_data = _get_monthly_candle_data(instrument_key)
        if not candle_data or not candle_data.candles:
            return f"❌ Could not fetch stock data for {company_query} (instrument: {instrument_key}). Please check if the company ticker is valid."
        
        # Step 3: Calculate price metrics
        price_metrics = _calculate_price_metrics(candle_data)
        if not price_metrics:
            return f"❌ Insufficient price data to analyze for {company_query}."
        
        emit_process({"message": f"Fetching recent news for {company_query}"})
        
        # Step 4: Get company news
        news_content = _get_company_news(company_query)
        
        emit_process({"message": f"Analyzing price-news correlation for {company_query}"})
        
        # Step 5: AI-powered correlation analysis
        correlation_analysis = _analyze_price_news_correlation(price_metrics, news_content, company_query)
        
        return correlation_analysis
        
    except Exception as e:
        logger.exception(f"Error in price movement analysis for {company_query}")
        return f"❌ Error during price movement analysis for {company_query}: {str(e)}"


# Create structured tool for use in subgraphs
from langchain.tools import StructuredTool

price_movement_tool = StructuredTool.from_function(
    func=price_movement_analysis,
    name="price_movement_analysis",
    description="Analyze stock price movements over the last month and correlate with news events using Upstox data and Exa news.",
    args_schema=PriceMovementInput,
)
