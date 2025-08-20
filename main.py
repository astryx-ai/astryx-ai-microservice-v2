import asyncio
import logging
from fastapi import FastAPI
from dotenv import load_dotenv

# Load .env early
load_dotenv()

from app.routes.chat import router as chat_router  # noqa: E402
from app.routes.rag import router as rag_router  # noqa: E402
from app.routes.health import router as health_router  # noqa: E402
from app.db import SessionLocal  # noqa: E402
from app.services.rag import background_refresh_loop  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai-microservice")

app = FastAPI(title="Astryx AI Microservice")
_stop_event: asyncio.Event | None = None
_bg_task: asyncio.Task | None = None


@app.get("/")
def read_root():
    return "Welcome to Astryx AI - AI Microservice"


app.include_router(chat_router)
app.include_router(rag_router)
app.include_router(health_router)


@app.on_event("startup")
async def _startup():
    global _stop_event, _bg_task
    _stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    _bg_task = loop.create_task(background_refresh_loop(SessionLocal, _stop_event))
    logger.info("Background refresh task started")


@app.on_event("shutdown")
async def _shutdown():
    global _stop_event, _bg_task
    if _stop_event is not None:
        _stop_event.set()
    if _bg_task is not None:
        try:
            await _bg_task
        except Exception:
            pass
