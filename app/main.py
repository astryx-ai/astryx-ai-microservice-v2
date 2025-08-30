from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.health import router as health_router
from app.routes.ingest_news import router as ingest_news_router
from app.routes.ingest_companies import router as ingest_companies_router
from app.routes.agent import router as agent_router
from app.routes.agent import stream_alias_router

app = FastAPI(title="Astryx AI Microservice")

# Health stays root
app.include_router(health_router)

# Add prefixes
# Canonical routes: /agent, /ingest/news, /ingest/companies, /health
app.include_router(agent_router)
app.include_router(stream_alias_router)
app.include_router(ingest_news_router)
app.include_router(ingest_companies_router)
