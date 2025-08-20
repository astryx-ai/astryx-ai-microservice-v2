import asyncio
import logging
from fastapi import FastAPI
from dotenv import load_dotenv

# Load .env early
load_dotenv()

from app.routes.chat import router as chat_router  # noqa: E402
from app.routes.health import router as health_router  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai-microservice")

app = FastAPI(title="Astryx AI Microservice")
_stop_event: asyncio.Event | None = None
_bg_task: asyncio.Task | None = None


@app.get("/")
def read_root():
    return "Welcome to Astryx AI - AI Microservice"


app.include_router(chat_router)
app.include_router(health_router)
