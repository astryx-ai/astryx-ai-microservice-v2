from fastapi import FastAPI
from app.routes.health import router as health_router
# from app.routes.ingest_news import router as ingest_news_router
from app.routes.ingest_companies import router as ingest_companies_router
from app.routes.agent import router as agent_router
from app.routes.stock_screen import router as stock_screen_router

app = FastAPI(title="Astryx AI Microservice")

# Health stays root
app.include_router(health_router)

# Add prefixes
app.include_router(agent_router)
# app.include_router(ingest_news_router)
app.include_router(ingest_companies_router)
app.include_router(stock_screen_router)
