from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from app.config import settings


def chat_model(temperature: float = 0.2):
    model_kwargs = {}

    return AzureChatOpenAI(
        azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT,
        openai_api_version=settings.AZURE_OPENAI_API_VERSION,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        openai_api_key=settings.AZURE_OPENAI_API_KEY,
        model_version="2024-11-20",
        temperature=temperature,
        streaming=True,
        model_kwargs=model_kwargs,
    )

def decision_model(temperature: float = 0.1):
    """Non-streaming, low-temperature model for routing/decisions."""
    model_kwargs = {}

    return AzureChatOpenAI(
        azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT,
        openai_api_version=settings.AZURE_OPENAI_API_VERSION,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        openai_api_key=settings.AZURE_OPENAI_API_KEY,
        model_version="2024-11-20",
        temperature=temperature,
        streaming=False,  # Explicitly disable streaming
        model_kwargs=model_kwargs,
    )


def embedder():
    dep = settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT or settings.AZURE_OPENAI_DEPLOYMENT
    return AzureOpenAIEmbeddings(
        azure_deployment=dep,
        openai_api_version=settings.AZURE_OPENAI_API_VERSION,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        openai_api_key=settings.AZURE_OPENAI_API_KEY,
    )
