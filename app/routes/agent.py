from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict
from starlette.responses import StreamingResponse
from typing import Optional, Any
import os
import json
import asyncio

from app.graph.runner import run_chat as run_chat_graph
from app.agents.super.runner import run_super_agent

try:
	# Optional memory store for clear operation; keep import lazy-friendly
	from app.tools.memory_store import global_memory_store
except Exception:  # pragma: no cover
	global_memory_store = None  # type: ignore

try:
	# Optional LC message classes for streaming; we degrade gracefully if missing
	from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore
except Exception:  # pragma: no cover
	HumanMessage = None  # type: ignore
	SystemMessage = None  # type: ignore


router = APIRouter(prefix="/agent", tags=["agent"])  # canonical route
# Alias router for frontend compatibility: exposes POST /stream/messages at root
stream_alias_router = APIRouter(prefix="")

# --- add helpers to avoid NameError and simple intent routing ---
def _is_memory_clear_query(text: str) -> bool:
    t = (text or "").strip().lower()
    return t in {"clear memory", "reset memory", "forget", "/clear", "/reset"}

def _is_chart_intent(text: str) -> bool:
    t = (text or "").lower()
    chart_keywords = ("chart", "graph", "plot", "visualize", "dashboard")
    return any(k in t for k in chart_keywords)

def _is_finance_intent(text: str) -> bool:
    t = (text or "").lower()
    finance_keywords = ("stock", "price", "ticker", "market", "portfolio", "finance")
    return any(k in t for k in finance_keywords)
# --- end helpers ---


class AgentStreamPayload(BaseModel):
    question: str | None = None
    query: str | None = None
    user_id: str | None = None
    chat_id: str | None = None


class AgentPayload(AgentStreamPayload):
    pass


@router.post("")
async def run_agent(payload: AgentPayload):
    """Non-streaming variant that returns a single JSON response.

    Useful for clients that can't or don't want to consume SSE.
    """
    try:
        text = (payload.question or payload.query or "").strip()
        thread_id = payload.chat_id or payload.user_id
        if not text:
            raise HTTPException(status_code=400, detail="query or question is required")

        # Memory clear check
        if _is_memory_clear_query(text):
            if global_memory_store:
                try:
                    global_memory_store().clear(chat_id=payload.chat_id, user_id=payload.user_id)
                except Exception:
                    pass
            return {"success": True, "data": {"response": "memory cleared"}}

        # Ensure graph chat pipeline enabled by default
        if os.getenv("GRAPH_CHAT_ENABLED") is None and os.getenv("GRAPH_ENABLED") is None:
            os.environ["GRAPH_CHAT_ENABLED"] = "true"

        response_str = ""
        chart_data = None
        charts = None

        if _is_chart_intent(text):
            loop = asyncio.get_running_loop()
            result_graph = await loop.run_in_executor(
                None,
                lambda: run_chat_graph({"query": text, "user_id": payload.user_id, "chat_id": payload.chat_id}),
            )
            response_str = result_graph.get("response", "") or ""
            chart_data = result_graph.get("chart_data")
            charts = result_graph.get("charts")
        elif _is_finance_intent(text):
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: run_super_agent(text, memory={}, thread_id=thread_id),
            )
            response_str = result.get("output", "") or ""
        else:
            loop = asyncio.get_running_loop()
            result_graph = await loop.run_in_executor(
                None,
                lambda: run_chat_graph({"query": text, "user_id": payload.user_id, "chat_id": payload.chat_id}),
            )
            response_str = result_graph.get("response", "") or ""
            chart_data = result_graph.get("chart_data")
            charts = result_graph.get("charts")

        return {"success": True, "data": {"response": response_str, "chart_data": chart_data, "charts": charts}}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def run_agent_stream_simple(payload: AgentStreamPayload):
    """Simple line-delimited streaming: emits JSON lines with {text, end, index}.

    Content-Type is text/plain to match clients that don't consume SSE.
    """
    try:
        text = (payload.question or payload.query or "").strip()
        thread_id = payload.chat_id or payload.user_id
        if not text:
            raise HTTPException(status_code=400, detail="query or question is required")

        if _is_memory_clear_query(text):
            if global_memory_store:
                try:
                    global_memory_store().clear(chat_id=payload.chat_id, user_id=payload.user_id)
                except Exception:
                    pass
            async def cleared():
                yield (json.dumps({"text": "memory cleared", "end": True, "index": 0}) + "\n").encode("utf-8")
            headers = {"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"}
            return StreamingResponse(cleared(), media_type="text/plain", headers=headers)

        if os.getenv("GRAPH_CHAT_ENABLED") is None and os.getenv("GRAPH_ENABLED") is None:
            os.environ["GRAPH_CHAT_ENABLED"] = "true"

        async def token_generator():
            idx = 0
            try:
                loop = asyncio.get_running_loop()
                response_str = ""
                chart_data = None
                charts = None
                if _is_chart_intent(text):
                    result_graph = await loop.run_in_executor(
                        None,
                        lambda: run_chat_graph({"query": text, "user_id": payload.user_id, "chat_id": payload.chat_id}),
                    )
                    response_str = result_graph.get("response", "") or ""
                    chart_data = result_graph.get("chart_data")
                    charts = result_graph.get("charts")
                elif _is_finance_intent(text):
                    result = await loop.run_in_executor(
                        None,
                        lambda: run_super_agent(text, memory={}, thread_id=thread_id),
                    )
                    response_str = result.get("output", "") or ""
                else:
                    result_graph = await loop.run_in_executor(
                        None,
                        lambda: run_chat_graph({"query": text, "user_id": payload.user_id, "chat_id": payload.chat_id}),
                    )
                    response_str = result_graph.get("response", "") or ""
                    chart_data = result_graph.get("chart_data")
                    charts = result_graph.get("charts")

                chunk = 200
                for i in range(0, len(response_str), chunk):
                    piece = response_str[i:i+chunk]
                    if piece:
                        yield (json.dumps({"text": piece, "end": False, "index": idx}) + "\n").encode("utf-8")
                        idx += 1
                if chart_data or charts:
                    yield (json.dumps({"charts": charts, "chart_data": chart_data, "end": False, "index": idx}) + "\n").encode("utf-8")
                    idx += 1
            except Exception as gen_err:
                # Include an error line then still close the stream
                yield (json.dumps({"error": str(gen_err), "end": False, "index": idx}) + "\n").encode("utf-8")
                idx += 1
            yield (json.dumps({"end": True, "index": idx}) + "\n").encode("utf-8")

        headers = {
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        }
        return StreamingResponse(token_generator(), media_type="text/plain", headers=headers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
