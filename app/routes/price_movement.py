"""
Price Movement API Routes
Test endpoint for the price movement analysis functionality.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging

from app.agent_tools.price_movement import price_movement_analysis

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/price-movement", tags=["price-movement"])


class PriceMovementRequest(BaseModel):
    """Request model for price movement analysis."""
    company_query: str
    instrument_key: Optional[str] = None


class PriceMovementResponse(BaseModel):
    """Response model for price movement analysis."""
    success: bool
    analysis: str
    error: Optional[str] = None


@router.post("/analyze", response_model=PriceMovementResponse)
async def analyze_price_movement(request: PriceMovementRequest):
    """
    Analyze stock price movements and correlate with news events.
    
    This endpoint demonstrates the price movement analysis functionality
    by fetching 1-month Upstox data and correlating it with Exa news.
    """
    try:
        logger.info(f"Price movement analysis requested for: {request.company_query}")
        
        # Call the price movement analysis tool
        analysis_result = price_movement_analysis.func(
            company_query=request.company_query,
            instrument_key=request.instrument_key
        )
        
        return PriceMovementResponse(
            success=True,
            analysis=analysis_result
        )
        
    except Exception as e:
        logger.error(f"Error in price movement analysis: {str(e)}")
        return PriceMovementResponse(
            success=False,
            analysis="",
            error=str(e)
        )


@router.get("/health")
async def health_check():
    """Health check endpoint for price movement service."""
    return {"status": "healthy", "service": "price-movement"}


@router.get("/supported-companies")
async def get_supported_companies():
    """Get list of supported companies for demo."""
    return {
        "supported_companies": [
            {"name": "Reliance Industries", "ticker": "RELIANCE", "instrument_key": "NSE_EQ|INE002A01018"},
            {"name": "Tata Consultancy Services", "ticker": "TCS", "instrument_key": "NSE_EQ|INE467B01029"},
            {"name": "Infosys", "ticker": "INFOSYS", "instrument_key": "NSE_EQ|INE009A01021"},
            {"name": "HDFC Bank", "ticker": "HDFC BANK", "instrument_key": "NSE_EQ|INE040A01034"},
            {"name": "ICICI Bank", "ticker": "ICICI BANK", "instrument_key": "NSE_EQ|INE090A01021"},
            {"name": "Bharti Airtel", "ticker": "BHARTI AIRTEL", "instrument_key": "NSE_EQ|INE397D01024"},
            {"name": "ITC", "ticker": "ITC", "instrument_key": "NSE_EQ|INE154A01025"},
            {"name": "State Bank of India", "ticker": "SBI", "instrument_key": "NSE_EQ|INE062A01020"},
            {"name": "Wipro", "ticker": "WIPRO", "instrument_key": "NSE_EQ|INE075A01022"},
            {"name": "Maruti Suzuki", "ticker": "MARUTI SUZUKI", "instrument_key": "NSE_EQ|INE585B01010"}
        ],
        "note": "These are demo mappings. In production, use a comprehensive instrument database."
    }
