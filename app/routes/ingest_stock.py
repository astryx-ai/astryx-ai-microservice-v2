from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from langchain.schema import Document
from app.services.rag import chunk_text, upsert_stocks
from app.scrapper.sanitize import clean_text
from app.scrapper.scrape_stock import get_stock_data  # your file

router = APIRouter(prefix="/ingest/stocks", tags=["ingest"])

class IngestStockPayload(BaseModel):
    ticker: str
    company: Optional[str] = None

def dict_to_text(data: Dict[str, Any]) -> str:
    # flatten dict into readable bullets/paragraph
    lines: List[str] = []
    def walk(prefix: str, val: Any):
        if isinstance(val, dict):
            for k,v in val.items():
                walk(f"{prefix}{k}.", v)
        elif isinstance(val, list):
            for i,v in enumerate(val):
                walk(f"{prefix}{i}.", v)
        else:
            if val is not None and str(val).strip():
                lines.append(f"{prefix} {val}")
    walk("", data)
    return "\n".join(lines)

@router.post("")
def ingest_stock(payload: IngestStockPayload):
    try:
        raw = get_stock_data(payload.ticker)
        if isinstance(raw, dict):
            text = dict_to_text(raw)
        else:
            text = str(raw or "")
        text = clean_text(text)
        if not text:
            return {"ingested": 0}

        meta = {
            "ticker": payload.ticker.upper(),
            "company": payload.company or "",
            "section": "fundamentals",
            "type": "fundamental"
        }
        docs = chunk_text(text, meta)
        upsert_stocks(docs)
        return {"ingested": len(docs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
