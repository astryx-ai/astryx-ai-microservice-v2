from __future__ import annotations
from functools import lru_cache
import logging
from typing import Optional
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv(override=False)


class Settings(BaseSettings):
    # OpenAI (public)
    OPENAI_API_KEY: Optional[str] = None

    # Azure OpenAI
    AZURE_OPENAI_API_KEY: Optional[str] = None
    AZURE_OPENAI_ENDPOINT: Optional[str] = None
    AZURE_OPENAI_DEPLOYMENT: Optional[str] = None
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: Optional[str] = None
    AZURE_OPENAI_API_VERSION: str = "2024-05-01-preview"

    # DB
    DATABASE_URL: str
    PGVECTOR_URL: Optional[str] = None

    # RAG
    REFRESH_MINUTES: int = 30
    RETRIEVER_TOP_K: int = 5

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def use_azure(self) -> bool:
        return bool(self.AZURE_OPENAI_API_KEY and self.AZURE_OPENAI_ENDPOINT)

    # Pydantic v2: use model_config only (no inner Config)


@lru_cache
def get_settings() -> Settings:
    return Settings()

# Log which provider is configured (best-effort; avoid import loops)
try:
    _settings = get_settings()
    _log = logging.getLogger("ai-microservice")
    _prov = "azure" if _settings.use_azure else ("openai" if _settings.OPENAI_API_KEY else "none")
    _log.info("Config: LLM provider=%s", _prov)
except Exception:
    pass
