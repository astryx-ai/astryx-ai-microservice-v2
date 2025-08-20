from typing import List, Dict, Any, Literal
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from .azure_openai import chat_model
from .vector_store import news_store, stock_store

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=150, separators=["\n\n", "\n", ". ", " "]
)

def chunk_text(text: str, metadata: Dict[str, Any]) -> List[Document]:
    return _splitter.create_documents([text], metadatas=[metadata])

def upsert_news(docs: List[Document]): news_store().add_documents(docs)
def upsert_stocks(docs: List[Document]): stock_store().add_documents(docs)

def retrieve_news(query: str, k=6, filters: Dict[str,str]|None=None):
    return news_store().similarity_search(query, k=k, filter=filters or {})

def retrieve_stocks(query: str, k=6, filters: Dict[str,str]|None=None):
    return stock_store().similarity_search(query, k=k, filter=filters or {})

_prompt = ChatPromptTemplate.from_template(
"""You are a precise markets analyst. Use the provided context to answer briefly.
If helpful, add bullet points and mention tickers inline.

Question:
{question}

Context:
{context}

Answer:"""
)

def rag_answer(question: str,
               domain: Literal["news","stocks","both"]="news",
               filters: Dict[str,str]|None=None) -> str:
    # gather context
    docs = []
    if domain in ("news","both"):
        docs += retrieve_news(question, k=4, filters=filters)
    if domain in ("stocks","both"):
        docs += retrieve_stocks(question, k=4, filters=filters)

    context = "\n\n".join(d.page_content for d in docs)
    chain = (_prompt | chat_model())
    resp = chain.invoke({"question": question, "context": context})
    return resp.content
