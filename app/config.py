from functools import lru_cache
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    # Azure OpenAI
    azure_openai_endpoint: str | None = Field(
        default=None, alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key: str | None = Field(
        default=None, alias="AZURE_OPENAI_API_KEY")
    azure_openai_deployment: str | None = Field(
        default=None, alias="AZURE_OPENAI_DEPLOYMENT")
    azure_openai_api_version: str = Field(
        default="2024-05-01-preview", alias="AZURE_OPENAI_API_VERSION")

    environment: str = Field(default="local", alias="ENVIRONMENT")

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore
