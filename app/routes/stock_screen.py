from fastapi import APIRouter, HTTPException, Query
from app.services.report import IngestorService
from typing import Optional
from datetime import datetime, timedelta
from app.services.upstox import UpstoxProvider
from app.utils.model import CandleQuery
from app.db import companies as companies_repo
from pydantic import BaseModel
from app.agent_tools.price_movement import price_movement_analysis


router = APIRouter(prefix="/stock-screen", tags=["stock-screen"])


@router.get("/fundamentals")
def get_stock_screen(
    stock_query: str = Query(..., description="Free-text company/stock query to resolve BSE scripcode")
):
    try:
        service = IngestorService()
        result = service.extract_fundamentals_header(stock_query=stock_query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _last_market_day(today: Optional[datetime] = None) -> datetime:
    d = (today or datetime.utcnow()).date()
    # roll back to last weekday (Mon=0..Sun=6)
    while d.weekday() > 4:  # 5=Sat, 6=Sun
        d = d - timedelta(days=1)
    return datetime(d.year, d.month, d.day)


def _resolve_isin_from_query(stock_query: str) -> Optional[str]:
    q = (stock_query or "").strip()
    if not q:
        return None
    
    print(f"Searching for ISIN for query: {q}")
    matches = companies_repo.search_companies(q, limit=1)
    if matches:
        isin = matches[0].get("isin")
        if isin:
            return str(isin)
    fuzzy = companies_repo.fuzzy_search_companies(q, limit=1)
    if fuzzy:
        isin = fuzzy[0].get("isin")
        if isin:
            return str(isin)
    return None


@router.get("/candle-history")
def get_candle_history(
    stock_query: str = Query(..., description="Free-text company/stock query to resolve ISIN for Upstox"),
    range: int = Query(..., ge=1, description="Number of days back from the latest market day"),
):
    try:
        isin = _resolve_isin_from_query(stock_query)
        if not isin:
            raise HTTPException(status_code=404, detail="Unable to resolve ISIN for the provided query")

        # Build Upstox instrument key: BSE_EQ%7C{ISIN}
        instrument_key = f"BSE_EQ%7C{isin}"

        latest_market_day = _last_market_day()
        to_date = latest_market_day.strftime("%Y-%m-%d")

        if range <= 5:
            # Intraday of the latest open day, 30-minute candles
            unit = "minutes"
            interval = "30"
            from_date = to_date
        else:
            # Daily candles for the last `range` days ending latest market day
            unit = "days"
            interval = "1"
            start_day = latest_market_day - timedelta(days=range)
            from_date = start_day.strftime("%Y-%m-%d")

        query = CandleQuery(
            instrument_key=instrument_key,
            unit=unit,
            interval=interval,
            from_date=from_date,
            to_date=to_date,
        )

        upstox = UpstoxProvider()
        raw = upstox.fetch_historical_candles(query)
        canonical = UpstoxProvider.convert_upstox_candles_to_canonical(raw, query)
        return canonical.dict()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class PriceMovementResponse(BaseModel):
    success: bool
    analysis: str
    error: Optional[str] = None


@router.get("/price-movement/analyze", response_model=PriceMovementResponse)
def analyze_price_movement(
    stock_query: str = Query(..., description="Free-text company/stock query to resolve ISIN for Upstox"),
):
    try:
        analysis_result = price_movement_analysis.func(
            company_query=stock_query
        )
        return PriceMovementResponse(success=True, analysis=analysis_result)
    except Exception as e:
        return PriceMovementResponse(success=False, analysis="", error=str(e)) 