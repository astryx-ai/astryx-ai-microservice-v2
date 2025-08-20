from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import List, Optional, Sequence

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models import Document, ChatMessage
from app.scrapper.scrape_stock import scrape_company as get_stock_data
from app.scrapper.scrape_news import scrape_company_news as get_news
from app.vectorstore import get_vector_service

from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.documents import Document as LCDocument

logger = logging.getLogger(__name__)
settings = get_settings()


def _flatten_stock_to_text(stock: dict) -> List[tuple[str, dict]]:
    texts: List[tuple[str, dict]] = []
    if not stock:
        return texts
    metrics = (stock or {}).get("metrics", {})
    header = f"Stock snapshot for {stock.get('company_name','')} as of {stock.get('as_of','')}\nSource: {stock.get('source_url','')}"
    lines = [header]
    for k, v in metrics.items():
        raw = v.get("raw_value")
        desc = v.get("description")
        lines.append(f"- {k}: {raw} ({desc})")
    full = "\n".join(lines)
    texts.append((full, {"kind": "stock", "source_url": stock.get("source_url")}))
    return texts


def _flatten_news_to_text(news: Sequence[dict]) -> List[tuple[str, dict]]:
    out: List[tuple[str, dict]] = []
    for item in news:
        title = item.get("title") or ""
        date = item.get("date") or ""
        url = item.get("url") or ""
        content = item.get("content") or ""
        text = f"Title: {title}\nDate: {date}\nURL: {url}\n\n{content}"
        out.append((text, {"kind": "news", "title": title, "date": date, "url": url}))
    return out


async def upsert_documents_for_symbol(session: AsyncSession, symbol: str) -> int:
    """Scrape stock + news for a symbol, store new Documents, and index to vector DB.
    Returns number of new docs added.
    """
    symbol_upper = symbol.upper()

    # Run scrapers in threadpool
    stock, news = await asyncio.gather(
        asyncio.to_thread(get_stock_data, symbol_upper),
        asyncio.to_thread(get_news, symbol_upper),
    )

    texts: List[tuple[str, dict]] = []
    texts += _flatten_stock_to_text(stock or {})
    texts += _flatten_news_to_text(news or [])

    # Insert new Document rows if not exists
    new_docs: List[Document] = []
    for text, meta in texts:
        # Check if a document with same identity already exists to avoid IntegrityError
        exists_q = select(Document.id).where(
            Document.symbol == symbol_upper,
            Document.source == (meta.get("kind", "unknown")),
            Document.doc_type == (meta.get("kind", "unknown")),
            Document.content == text,
        ).limit(1)
        exists = (await session.execute(exists_q)).first()
        if exists:
            continue
        doc = Document(
            symbol=symbol_upper,
            source=meta.get("kind", "unknown"),
            doc_type=meta.get("kind", "unknown"),
            content=text,
            meta=meta,
        )
        session.add(doc)
        await session.flush()  # assign id
        new_docs.append(doc)

    if new_docs:
        await session.commit()
        # Index to vector store
        vector_service = get_vector_service()
        contents = [d.content for d in new_docs]
        metadatas = [{"id": d.id, "symbol": d.symbol, **(d.meta or {})} for d in new_docs]
        # Chunk contents and index
        chunk_texts: List[str] = []
        chunk_metas: List[dict] = []
        for text, meta in zip(contents, metadatas):
            chunks = vector_service.chunk(text, meta)
            for ch in chunks:
                chunk_texts.append(ch.page_content)
                chunk_metas.append(ch.metadata)
        await asyncio.to_thread(vector_service.add_texts, chunk_texts, chunk_metas)

    return len(new_docs)


async def retrieve_context(query: str, symbol: str, k: Optional[int] = None) -> List[LCDocument]:
    k = k or getattr(settings, "RETRIEVER_TOP_K", 6)
    # Use filter by symbol
    vector_service = get_vector_service()
    results = await asyncio.to_thread(vector_service.vstore.similarity_search, query, k, {"symbol": symbol.upper()})
    return results


async def answer_query(session: AsyncSession, query: str, symbol: str, user_id: Optional[str] = None, chat_id: Optional[str] = None) -> dict:
    # Fetch chat history
    stmt = select(ChatMessage).where(ChatMessage.chat_id == (chat_id or "")).order_by(ChatMessage.created_at.desc()).limit(10)
    history = (await session.execute(stmt)).scalars().all() if chat_id else []

    # Retrieve context
    docs = await retrieve_context(query, symbol)
    context_text = "\n\n".join([f"[Doc {i+1}] {d.page_content[:1200]}" for i, d in enumerate(docs)])

    system_prompt = (
        "You are a helpful financial assistant. Answer using only the provided CONTEXT and your general knowledge. "
        "Cite document numbers like [Doc 1] when relevant. If uncertain, say you are not sure."
    )

    history_msgs = []
    for m in reversed(history):  # oldest first
        history_msgs.append({"role": m.role, "content": m.content})

    messages = (
        [{"role": "system", "content": system_prompt}] + history_msgs +
        [
            {"role": "user", "content": f"CONTEXT:\n{context_text}\n\nUSER QUESTION: {query}"}
        ]
    )

    if settings.use_azure:
        llm = AzureChatOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT or "gpt-4o",
            api_version=settings.AZURE_OPENAI_API_VERSION,
            temperature=0.2,
        )
    else:
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=settings.OPENAI_API_KEY, temperature=0.2)
    resp = await asyncio.to_thread(llm.invoke, messages)
    answer = getattr(resp, "content", "") if resp else ""

    # Persist chat messages and index them for future RAG
    if chat_id and user_id:
        user_msg = ChatMessage(chat_id=chat_id, user_id=user_id, role="user", content=query)
        asst_msg = ChatMessage(chat_id=chat_id, user_id=user_id, role="assistant", content=answer)
        session.add_all([user_msg, asst_msg])
        await session.commit()
        # Vectorize
        vector_service = get_vector_service()
        pairs = [(user_msg, {"kind": "chat", "role": "user"}), (asst_msg, {"kind": "chat", "role": "assistant"})]
        chunk_texts: List[str] = []
        chunk_metas: List[dict] = []
        for msg, meta in pairs:
            # Also store as Document rows for dedup/history; ignore conflicts
            try:
                doc = Document(symbol=symbol.upper(), source="chat", doc_type="message", content=msg.content, meta={"chat_id": chat_id, "role": msg.role})
                session.add(doc)
                await session.flush()
            except Exception:
                await session.rollback()
            for ch in vector_service.chunk(msg.content, {"symbol": symbol.upper(), "chat_id": chat_id, **meta}):
                chunk_texts.append(ch.page_content)
                chunk_metas.append(ch.metadata)
        if chunk_texts:
            await asyncio.to_thread(vector_service.add_texts, chunk_texts, chunk_metas)

    return {
        "answer": answer,
        "sources": [
            {"metadata": d.metadata} for d in docs
        ],
    }


async def distinct_symbols(session: AsyncSession) -> List[str]:
    res = await session.execute(select(Document.symbol).distinct())
    return [r[0] for r in res.fetchall()]


async def background_refresh_loop(session_factory, stop_event: asyncio.Event):
    interval = max(1, getattr(settings, "REFRESH_MINUTES", 30)) * 60
    logger.info("Starting background refresh loop with %s minutes", getattr(settings, "REFRESH_MINUTES", 30))
    while not stop_event.is_set():
        try:
            async with session_factory() as s:  # type: ignore
                for symbol in await distinct_symbols(s):
                    try:
                        added = await upsert_documents_for_symbol(s, symbol)
                        logger.info("Refreshed %s; new docs: %s", symbol, added)
                    except Exception as e:
                        logger.exception("Refresh error for %s: %s", symbol, e)
        except Exception as e:
            logger.exception("Refresh loop iteration error: %s", e)
        # sleep with cancellation support
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            continue
