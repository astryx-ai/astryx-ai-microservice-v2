from __future__ import annotations

import logging
from typing import Iterable, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings

# LangChain imports
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LCDocument
from langchain_postgres import PGVector

logger = logging.getLogger(__name__)
settings = get_settings()


def _pgvector_conn_str() -> str:
    url = settings.PGVECTOR_URL or settings.DATABASE_URL
    if url.startswith("postgresql+asyncpg://"):
        url = url.replace("postgresql+asyncpg://", "postgresql+psycopg://", 1)
    elif url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def _embeddings():
    if settings.use_azure:
        return AzureOpenAIEmbeddings(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            azure_deployment=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT or "text-embedding-3-small",
            api_version=settings.AZURE_OPENAI_API_VERSION,
        )
    if not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not configured and Azure not enabled")
    return OpenAIEmbeddings(model="text-embedding-3-small", api_key=settings.OPENAI_API_KEY)


class VectorStoreService:
    def __init__(self):
        self.settings = settings
        if not self.settings.DATABASE_URL:
            raise RuntimeError("DATABASE_URL not configured")
        emb = _embeddings()
        conn = _pgvector_conn_str()
        # PGVector will create table if not exist. Ignore extension creation failures on managed DBs.
        try:
            self.vstore = PGVector(
                embeddings=emb,
                collection_name=getattr(self.settings, "pgvector_collection", "ai_microservice"),
                connection=conn,
                use_jsonb=True,
                create_extension=True,
            )
        except Exception:
            logger.warning("PGVector init with create_extension failed; retrying without extension creation")
            self.vstore = PGVector(
                embeddings=emb,
                collection_name=getattr(self.settings, "pgvector_collection", "ai_microservice"),
                connection=conn,
                use_jsonb=True,
                create_extension=False,
            )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=getattr(self.settings, "max_chunk_size", 800),
            chunk_overlap=getattr(self.settings, "chunk_overlap", 100),
        )

    def chunk(self, text: str, metadata: Optional[dict] = None) -> List[LCDocument]:
        docs = [LCDocument(page_content=text, metadata=metadata or {})]
        return self.splitter.split_documents(docs)

    async def add_texts(self, texts: Iterable[str], metadatas: Optional[Iterable[dict]] = None) -> List[str]:
        # langchain-postgres add_texts is sync; call via thread if needed. Here we call sync.
        return self.vstore.add_texts(list(texts), metadatas=list(metadatas) if metadatas else None)

    def as_retriever(self):
        return self.vstore.as_retriever(search_kwargs={"k": getattr(self.settings, "retriever_top_k", 6)})

_singleton: VectorStoreService | None = None


def get_vector_service() -> VectorStoreService:
    global _singleton
    if _singleton is None:
        _singleton = VectorStoreService()
    return _singleton
