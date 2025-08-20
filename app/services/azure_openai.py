from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from .config import settings


def chat_model(temperature: float = 0.2):
    return AzureChatOpenAI(
        azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT,
        openai_api_version=settings.AZURE_OPENAI_API_VERSION,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        openai_api_key=settings.AZURE_OPENAI_API_KEY,
        temperature=temperature,
    )


def embedder():
    dep = settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT or settings.AZURE_OPENAI_DEPLOYMENT
    return AzureOpenAIEmbeddings(
        azure_deployment=dep,
        openai_api_version=settings.AZURE_OPENAI_API_VERSION,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        openai_api_key=settings.AZURE_OPENAI_API_KEY,
    )
