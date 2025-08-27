from typing import List, Dict, Any
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .vector_store import news_store, stock_store

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=150, separators=["\n\n", "\n", ". ", " "]
)


def chunk_text(text: str, metadata: Dict[str, Any]) -> List[Document]:
    text = (text or "").strip()
    if not text:
        return []
    chunks = _splitter.split_text(text)
    docs = [Document(page_content=chunk, metadata=dict(metadata or {})) for chunk in chunks]
    return docs


def upsert_news(docs: List[Document]):
    try:
        if not docs:
            return
        seen = set()
        uniq: List[Document] = []
        for d in docs:
            md = d.metadata or {}
            url = (md.get("url") or md.get("source") or "").strip().lower()
            title = (md.get("title") or "").strip().lower()
            key = (url, title)
            if key not in seen:
                seen.add(key)
                uniq.append(d)
        if not uniq:
            return
        news_store().add_documents(uniq)
        print(f"[RAG] upsert_news: inserted={len(uniq)} skipped={len(docs)-len(uniq)}")
    except Exception as e:
        print(f"[RAG] upsert_news failed: {e}")


def upsert_stocks(docs: List[Document]):
    try:
        if not docs:
            return
        seen = set()
        uniq: List[Document] = []
        for d in docs:
            md = d.metadata or {}
            key = ((md.get("ticker") or "").upper(), md.get("ts"))
            if key not in seen:
                seen.add(key)
                uniq.append(d)
        if not uniq:
            return
        stock_store().add_documents(uniq)
        print(f"[RAG] upsert_stocks: inserted={len(uniq)} skipped={len(docs)-len(uniq)}")
    except Exception as e:
        print(f"[RAG] upsert_stocks failed: {e}")
