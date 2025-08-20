from __future__ import annotations

from fastapi import APIRouter
from app.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    s = get_settings()
    provider = "azure" if s.use_azure else ("openai" if s.OPENAI_API_KEY else "none")
    return {
        "provider": provider,
        "azure_deploy": s.AZURE_OPENAI_DEPLOYMENT,
        "azure_embed": s.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        "db_url_scheme": (s.DATABASE_URL.split("://", 1)[0] if getattr(s, "DATABASE_URL", None) else None),
    }
