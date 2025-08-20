from functools import lru_cache
from pydantic import Field
try:
    from pydantic_settings import BaseSettings, SettingsConfigDict  # type: ignore
except ImportError:  # pragma: no cover
    # Fallback in case dependency missing – avoids immediate crash but .env loading disabled
    from pydantic import BaseModel as BaseSettings  # type: ignore

    class SettingsConfigDict(dict):  # type: ignore
        pass


class Settings(BaseSettings):
    """Application settings loaded from environment / .env.

    Field names are lowercase; environment variables use the UPPERCASE aliases.
    """
    azure_openai_endpoint: str | None = Field(
        default=None, alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key: str | None = Field(
        default=None, alias="AZURE_OPENAI_API_KEY")
    azure_openai_deployment: str | None = Field(
        default=None, alias="AZURE_OPENAI_DEPLOYMENT")
    azure_openai_api_version: str = Field(
        default="2024-05-01-preview", alias="AZURE_OPENAI_API_VERSION")
    azure_openai_embedding_deployment: str | None = Field(
        default=None, alias="AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

    # Generic OpenAI (non-Azure) fallback
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")

    # Database (e.g., Postgres w/ pgvector)
    database_url: str | None = Field(default=None, alias="DATABASE_URL")

    environment: str = Field(default="local", alias="ENVIRONMENT")

    @property
    def use_azure(self) -> bool:
        return bool(self.azure_openai_api_key and self.azure_openai_endpoint and self.azure_openai_deployment)

    @property
    def use_openai(self) -> bool:
        return bool(self.openai_api_key and not self.use_azure)

    # Pydantic v2 settings config
    try:
        model_config = SettingsConfigDict(
            # type: ignore[attr-defined]
            env_file=".env", case_sensitive=False)
    except Exception:  # pragma: no cover
        pass


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore
