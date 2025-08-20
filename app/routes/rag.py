from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse

from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db, SessionLocal
from app.services.rag import upsert_documents_for_symbol, answer_query
from app.schemas import AskRequest, AskResponse, CandleResponse

import yfinance as yf

router = APIRouter(tags=["rag"])


@router.post("/scrape/{symbol}")
async def scrape_symbol(symbol: str, db: Annotated[AsyncSession, Depends(get_db)]):
    added = await upsert_documents_for_symbol(db, symbol)
    return {"success": True, "symbol": symbol.upper(), "new_docs": added}


@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest, db: Annotated[AsyncSession, Depends(get_db)]):
    try:
        result = await answer_query(db, req.query, req.symbol, req.user_id, req.chat_id)
        return AskResponse(success=True, answer=result["answer"], sources=result["sources"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _ohlcv(symbol: str, days: int = 30):
    end = datetime.now()
    start = end - timedelta(days=days)
    df = yf.download(symbol, start=start, end=end, interval="1d", progress=False)
    if df is None or df.empty:
        return []
    df.reset_index(inplace=True)
    out = []
    for _, row in df.iterrows():
        out.append({
            "date": row["Date"].isoformat() if hasattr(row["Date"], 'isoformat') else str(row["Date"]),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": float(row["Volume"]) if row.get("Volume") is not None else None
        })
    return out


@router.get("/candles/{symbol}")
async def candles(symbol: str, days: int = Query(30, ge=1, le=365), stream: bool = Query(False)):
    if not stream:
        data = await asyncio.to_thread(_ohlcv, symbol, days)
        return CandleResponse(symbol=symbol.upper(), days=days, candles=data)

    async def event_gen():
        data = await asyncio.to_thread(_ohlcv, symbol, days)
        for item in data:
            yield f"data: {item}\n\n"
            await asyncio.sleep(0.01)

    return StreamingResponse(event_gen(), media_type="text/event-stream")
