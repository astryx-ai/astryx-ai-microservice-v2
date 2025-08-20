from __future__ import annotations

from fastapi import APIRouter
from app.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    s = get_settings()
    provider = "azure" if s.use_azure else (
        "openai" if s.use_openai else "none")
    return {
        "provider": provider,
        "azure_deploy": s.azure_openai_deployment,
        "azure_embed": s.azure_openai_embedding_deployment,
        "db_url_scheme": (s.database_url.split("://", 1)[0] if s.database_url else None),
    }
