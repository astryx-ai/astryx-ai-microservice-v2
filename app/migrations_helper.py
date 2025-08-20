from __future__ import annotations

import asyncio
from app.db import engine, Base
import app.models  # noqa: F401 - ensure model tables are registered


async def run():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


if __name__ == "__main__":
    asyncio.run(run())
