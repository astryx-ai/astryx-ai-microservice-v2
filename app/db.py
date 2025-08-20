from __future__ import annotations

import logging
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base

from app.config import get_settings

logger = logging.getLogger(__name__)

Base = declarative_base()


def _build_async_url(sync_url: str) -> str:
    # Accepts sync or async URL; normalize to asyncpg driver
    # Examples:
    # - postgresql://user:pass@host/db -> postgresql+asyncpg://user:pass@host/db
    # - postgresql+psycopg2://... -> postgresql+asyncpg://...
    if sync_url.startswith("postgresql+asyncpg://"):
        return sync_url
    if sync_url.startswith("postgres://"):
        sync_url = sync_url.replace("postgres://", "postgresql://", 1)
    if sync_url.startswith("postgresql+"):
        return "postgresql+asyncpg://" + sync_url.split("://", 1)[1]
    if sync_url.startswith("postgresql://"):
        return sync_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return sync_url


def make_engine_and_session() -> tuple:
    settings = get_settings()
    if not settings.DATABASE_URL:
        raise RuntimeError("DATABASE_URL not configured in .env")
    async_url = _build_async_url(settings.DATABASE_URL)
    engine = create_async_engine(async_url, pool_size=10, max_overflow=20, pool_pre_ping=True)
    session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    return engine, session_maker


engine, SessionLocal = make_engine_and_session()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    session: AsyncSession = SessionLocal()
    try:
        yield session
    finally:
        await session.close()
