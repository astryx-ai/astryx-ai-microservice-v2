from __future__ import annotations
from typing import Dict, Any
from langchain.prompts import ChatPromptTemplate
from app.tools.azure_openai import chat_model


def casual_response(question: str) -> str:
    prompt = ChatPromptTemplate.from_template(
        """You're a friendly assistant for a stocks+news app. Keep answers short (1-2 sentences).
If the question is about features, replies should be factual about this app.
User: {q}
Answer:"""
    )
    try:
        res = (prompt | chat_model(temperature=0.3)).invoke({"q": question or ""})
        return (res.content or "").strip()
    except Exception:
        return "Hi! Ask me about NSE/BSE stocks, charts, or company news."
