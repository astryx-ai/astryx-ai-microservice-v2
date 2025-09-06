import requests
from typing import Any, Dict, Optional, List

from app.config import settings
from app.utils.model import CandleQuery, Candle, CandleSeries


class UpstoxProvider:
    """Low-level client for Upstox market data calls."""
    base_url = "https://api.upstox.com/v3"

    def __init__(self, api_key: Optional[str] = None, secret: Optional[str] = None):
        self.api_key = api_key
        self.secret = secret

    def fetch_historical_candles(self, query: CandleQuery) -> Dict[str, Any]:
        url = f"{self.base_url}/historical-candle/{query.instrument_key}/{query.unit}/{query.interval}/{query.to_date}/{query.from_date}"
        headers = {"Accept": "application/json"}
        # If available, include OAuth access token; else include API key header if that's what your plan requires.
        if settings.UPSTOX_ACCESS_TOKEN:
            headers["Authorization"] = f"Bearer {settings.UPSTOX_ACCESS_TOKEN}"
        elif settings.UPSTOX_API_KEY:
            headers["x-api-key"] = settings.UPSTOX_API_KEY
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json() 
    

    def convert_upstox_candles_to_canonical(raw: Dict[str, Any], query: CandleQuery) -> CandleSeries:
        """Convert Upstox historical candle response to canonical CandleSeries."""
        candles_list: List[List] = raw.get("data", {}).get("candles", [])
        candles: List[Candle] = []
        for item in candles_list:
            if not isinstance(item, list) or len(item) < 5:
                continue
            # Upstox order: [timestamp, open, high, low, close, volume?]
            timestamp = str(item[0])
            open_price = float(item[1])
            high_price = float(item[2])
            low_price = float(item[3])
            close_price = float(item[4])
            volume = float(item[5]) if len(item) > 5 and item[5] is not None else None
            candles.append(Candle(
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume,
                timestamp=timestamp
            ))
        return CandleSeries(
            instrument_key=query.instrument_key,
            unit=query.unit,
            interval=query.interval,
            from_date=query.from_date,
            to_date=query.to_date,
            candles=candles,
            provider="upstox"
        ) 