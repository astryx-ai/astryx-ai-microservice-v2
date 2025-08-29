from fastapi import FastAPI
from app.routes.health import router as health_router
from app.routes.ingest_news import router as ingest_news_router
from app.routes.chat import router as chat_router
from app.routes.ingest_companies import router as ingest_companies_router
from app.routes.chat import agent_router, super_router

app = FastAPI(title="Astryx AI Microservice")

# Health stays root
app.include_router(health_router)

# Add prefixes
# Canonical routes: /chat, /ingest/news, /ingest/companies, /health
# Deprecated shims (kept for compat with warnings): /agent, /super/chat, /super/memory/clear
app.include_router(chat_router)
app.include_router(agent_router)  # deprecated shim
app.include_router(super_router)  # deprecated shim
app.include_router(ingest_news_router)
app.include_router(ingest_companies_router)
