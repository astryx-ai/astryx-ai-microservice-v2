from pydantic_settings import BaseSettings
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# Proactively load .env so settings are present even if CWD varies
# 1) Try to find via cwd walk-up
_dotenv_path = find_dotenv(filename=".env", usecwd=True)
# 2) Fallback to repo root relative to this file (../.env)
if not _dotenv_path:
    repo_root = Path(__file__).resolve().parents[1]
    candidate = repo_root / ".env"
    if candidate.exists():
        _dotenv_path = str(candidate)
# Load if found (don't override already-set envs)
if _dotenv_path:
    load_dotenv(_dotenv_path, override=False)


class Settings(BaseSettings):
    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_DEPLOYMENT: str
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str | None = None
    AZURE_OPENAI_API_VERSION: str = "2024-05-01-preview"

    SUPABASE_URL: str
    SUPABASE_SERVICE_KEY: str

    DATABASE_URL: str | None = None

    ENVIRONMENT: str = "local"

    EXA_API_KEY: str

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()


