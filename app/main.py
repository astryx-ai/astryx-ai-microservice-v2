from fastapi import FastAPI
from app.routes.health import router as health_router
from app.routes.ingest_news import router as ingest_news_router
from app.routes.ingest_stock import router as ingest_stock_router
from app.routes.chat import router as chat_router
from app.routes.candles import router as candles_router

app = FastAPI(title="Markets AI Microservice")

# Health stays root
app.include_router(health_router)

# Add prefixes
app.include_router(ingest_news_router)
app.include_router(ingest_stock_router)
app.include_router(chat_router)
app.include_router(candles_router)
