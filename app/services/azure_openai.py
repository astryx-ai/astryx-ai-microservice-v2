"""Azure OpenAI service wrapper."""
from __future__ import annotations

try:
    from openai import AzureOpenAI  # type: ignore
except ImportError:  # pragma: no cover
    AzureOpenAI = None  # type: ignore

from app.config import get_settings


class AzureOpenAIService:
    def __init__(self):
        self.settings = get_settings()
        self._client = self._init_client()

    def _init_client(self):  # type: ignore
        if AzureOpenAI is None:
            return None
        if not self.settings.AZURE_OPENAI_ENDPOINT or not self.settings.AZURE_OPENAI_API_KEY:
            return None
        return AzureOpenAI(
            api_key=self.settings.AZURE_OPENAI_API_KEY,
            api_version=self.settings.AZURE_OPENAI_API_VERSION,
            azure_endpoint=self.settings.AZURE_OPENAI_ENDPOINT.rstrip(
                "/") + "/",
        )

    def is_configured(self) -> bool:
        return (
            self._client is not None
            and self.settings.AZURE_OPENAI_DEPLOYMENT is not None
        )

    async def chat(self, prompt: str) -> tuple[str, int]:
        if not self.is_configured():
            raise RuntimeError("Azure OpenAI not configured")

        completion = self._client.chat.completions.create(  # type: ignore
            model=self.settings.AZURE_OPENAI_DEPLOYMENT,  # type: ignore
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=512,
        )
        message_content = completion.choices[0].message.content if getattr(
            completion, "choices", []) else ""
        usage = getattr(completion, "usage", None)
        total_tokens = getattr(usage, "total_tokens", 0) if usage else 0
        return message_content or "", total_tokens


azure_openai_service = AzureOpenAIService()
