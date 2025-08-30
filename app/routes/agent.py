from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict
from starlette.responses import StreamingResponse
from typing import Optional, Any
import os
import json

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


class AgentStreamPayload(BaseModel):
    question: str
    user_id: str | None = None
    chat_id: str | None = None


@router.post("/stream")
async def run_agent_stream(payload: AgentStreamPayload):
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

            async def cleared():
                yield (
                    json.dumps({"success": True, "data": {"response": "memory cleared"}}) + "\n"
                ).encode("utf-8")

            return StreamingResponse(cleared(), media_type="application/json")

        # Ensure graph chat pipeline enabled
        if os.getenv("GRAPH_CHAT_ENABLED") is None and os.getenv("GRAPH_ENABLED") is None:
            os.environ["GRAPH_CHAT_ENABLED"] = "true"

        async def generator():
            # Run same logic as /agent
            response_str = ""
            chart_data = None
            charts = None

            if _is_chart_intent(text):
                result_graph = run_chat_graph(
                    {"query": text, "user_id": payload.user_id, "chat_id": payload.chat_id}
                )
                response_str = result_graph.get("response", "") or ""
                chart_data = result_graph.get("chart_data")
                charts = result_graph.get("charts")

            elif _is_finance_intent(text):
                result = run_super_agent(text, memory={}, thread_id=thread_id)
                response_str = result.get("output", "") or ""

            else:
                result_graph = run_chat_graph(
                    {"query": text, "user_id": payload.user_id, "chat_id": payload.chat_id}
                )
                response_str = result_graph.get("response", "") or ""
                chart_data = result_graph.get("chart_data")
                charts = result_graph.get("charts")

            # Now stream it out in chunks (so frontend can consume progressively)
            chunk_size = 200
            for i in range(0, len(response_str), chunk_size):
                piece = response_str[i : i + chunk_size]
                yield (
                    json.dumps(
                        {"success": True, "data": {"response": piece, "end": False, "index": i // chunk_size}}
                    )
                    + "\n"
                ).encode("utf-8")

            # If charts exist, stream them at the end
            if chart_data or charts:
                yield (
                    json.dumps(
                        {"success": True, "data": {"chart_data": chart_data, "charts": charts}}
                    )
                    + "\n"
                ).encode("utf-8")

            # Final marker
            yield (json.dumps({"end": True}) + "\n").encode("utf-8")

        headers = {
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        }
        return StreamingResponse(generator(), media_type="application/json", headers=headers)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# Frontend compatibility alias: POST /stream/messages → same as /agent/stream
@stream_alias_router.post("/stream/messages")
async def run_stream_messages_alias(payload: AgentStreamPayload):
	return await run_agent_stream(payload)
