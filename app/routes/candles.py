from fastapi import APIRouter, HTTPException
from typing import Literal
import yfinance as yf

router = APIRouter(prefix="/candles", tags=["market"])

@router.get("")
def candles(
    ticker: str,
    period: str = "1mo",
    interval: str = "1d",
):
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data is None or data.empty:
            return {"ticker": ticker.upper(), "candles": []}
        data = data.reset_index()
        out = []
        # yfinance uses "Datetime" for intraday, "Date" for daily
        tcol = "Datetime" if "Datetime" in data.columns else "Date"
        for _, row in data.iterrows():
            out.append({
                "t": str(row[tcol]),
                "o": float(row["Open"]),
                "h": float(row["High"]),
                "l": float(row["Low"]),
                "c": float(row["Close"]),
                "v": float(row["Volume"]),
            })
        return {"ticker": ticker.upper(), "period": period, "interval": interval, "candles": out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
