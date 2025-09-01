# from fastapi import APIRouter, HTTPException
# from pydantic import BaseModel
# from typing import Optional, List, Dict, Any
# from langchain.schema import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from app.scrapper.sanitize import clean_text
# from app.scrapper.scrape_news import get_news
# from app.services.tools.rag_tools import news_store


# router = APIRouter(prefix="/ingest/news", tags=["ingest"])


# _splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000, chunk_overlap=150, separators=["\n\n", "\n", ". ", " "]
# )
# def chunk_text(text: str, metadata: Dict[str, Any]) -> List[Document]:
#     return _splitter.create_documents([text], metadatas=[metadata])


# def upsert_news(docs: List[Document]): news_store().add_documents(docs)

# def retrieve_news(query: str, k=6, filters: Dict[str, str] | None = None):
#     return news_store().similarity_search(query, k=k, filter=filters or {})

# class IngestNewsPayload(BaseModel):
#     ticker: str
#     company: Optional[str] = None
#     limit: int = 20


# @router.post("")
# def ingest_news(payload: IngestNewsPayload):
#     try:
#         items = get_news(payload.ticker) or []
#         items = items[: payload.limit]
#         docs: List[Document] = []
#         for it in items:
#             title = it.get("title", "")
#             text = it.get("text") or it.get("summary") or ""
#             blob = clean_text(f"{title}. {text}")
#             if not blob:
#                 continue
#             meta: Dict[str, Any] = {
#                 "ticker": payload.ticker.upper(),
#                 "company": payload.company or "",
#                 "source": it.get("url") or it.get("source") or "news",
#                 "title": title,
#                 "type": "news"
#             }
#             docs.extend(chunk_text(blob, meta))
#         if docs:
#             upsert_news(docs)
#         return {"ingested": len(docs)}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
