from pydantic_settings import BaseSettings, SettingsConfigDict

# No explicit load_dotenv. Rely on Pydantic to read .env from the repo root.


class Settings(BaseSettings):
    # Pydantic Settings v2 configuration — ensures .env is read even if CWD varies
    model_config = SettingsConfigDict(
    env_file=".env",
        extra="ignore",
    )

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


settings = Settings()


