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
        if not self.settings.azure_openai_endpoint or not self.settings.azure_openai_api_key:
            return None
        return AzureOpenAI(
            api_key=self.settings.azure_openai_api_key,
            api_version=self.settings.azure_openai_api_version,
            azure_endpoint=self.settings.azure_openai_endpoint.rstrip(
                "/") + "/",
        )

    def is_configured(self) -> bool:
        return (
            self._client is not None
            and self.settings.azure_openai_deployment is not None
        )

    async def chat(self, prompt: str) -> tuple[str, int]:
        if not self.is_configured():
            raise RuntimeError("Azure OpenAI not configured")

        completion = self._client.chat.completions.create(  # type: ignore
            model=self.settings.azure_openai_deployment,  # type: ignore
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
