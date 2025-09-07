from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    AZURE_OPENAI_ENDPOINT: str = "https://openaitest000111.openai.azure.com/"
    AZURE_OPENAI_API_KEY: str = "280f844586fe49c3ad3706cbf1868a0c"
    AZURE_OPENAI_DEPLOYMENT: str = "gpt-4o"
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str | None = None
    AZURE_OPENAI_API_VERSION: str = "2025-01-01-preview"

    SUPABASE_URL: str = "https://evuduhnqajzrxjczmeow.supabase.co"
    SUPABASE_SERVICE_KEY: str | None = None
    SUPABASE_ANON_KEY: str = "sb_publishable_U1M9e0YpdtKYNUK6QrpOXw_iPgOKEjk"

    DATABASE_URL: str = "postgresql://postgres.evuduhnqajzrxjczmeow:cUU8YAXxuxVVcBZh@aws-0-ap-south-1.pooler.supabase.com:5432/postgres"

    ENVIRONMENT: str = "local"

    EXA_API_KEY: str = "3e18d5f9-4324-4bdd-8ea7-3142afeb4daa"
    
    # Upstox API settings
    UPSTOX_API_KEY: str | None = None
    UPSTOX_ACCESS_TOKEN: str | None = None
    
    # Additional OpenAI settings
    OPENAI_API_KEY: str = "280f844586fe49c3ad3706cbf1868a0c"
    
    # LangSmith settings
    LANGSMITH_TRACING: bool = True
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGSMITH_API_KEY: str = "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"
    LANGSMITH_PROJECT: str = "pr-ajar-series-72"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()


