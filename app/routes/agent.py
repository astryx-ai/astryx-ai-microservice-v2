from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict
from starlette.responses import StreamingResponse
from typing import Optional, Any
import os
import json

from app.graph.runner import run_chat as run_chat_graph
from app.agents.super.runner import run_super_agent
from app.services.agent import build_agent

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


class ChatPayload(BaseModel):
	model_config = ConfigDict(extra="ignore")
	query: Optional[str] = None
	question: Optional[str] = None
	user_id: Optional[str] = None
	chat_id: Optional[str] = None
	reset_memory: Optional[bool] = False
	memory_clear: Optional[bool] = False


class ChatResponseData(BaseModel):
	response: str
	chart_data: Optional[dict] = None
	charts: Optional[list] = None
	tokens_used: Optional[int] = None
	cost: Optional[float] = None


class ChatResponse(BaseModel):
	success: bool
	data: ChatResponseData


def _is_memory_clear_query(q: str) -> bool:
	qn = (q or "").strip().lower()
	return qn in {"clear memory", "reset memory", "reset chat", "clear conversation", "forget"}


def _is_chart_intent(q: str) -> bool:
	qn = (q or "").lower()
	for k in ("chart", "candle", "candlestick", "ohlc", "line chart", "area chart", "bar chart", "price chart"):
		if k in qn:
			return True
	return False


def _is_finance_intent(q: str) -> bool:
	qn = (q or "").lower()
	finance_words = [
		"stock", "share", "ticker", "nse", "bse", "price", "market cap", "pe", "fundamental", "intraday",
		"earnings", "revenue", "profit", "dividend", "company", "quarter", "results",
	]
	return any(w in qn for w in finance_words)


@router.post("")
def agent_chat(payload: ChatPayload) -> Any:
	try:
		# Normalize inputs across legacy/merged routes
		text = (payload.query or payload.question or "").strip()
		thread_id = payload.chat_id or payload.user_id

		# Memory clear operation: parameter-based OR natural-language trigger
		if payload.memory_clear or _is_memory_clear_query(text):
			if global_memory_store:
				try:
					global_memory_store().clear(chat_id=payload.chat_id, user_id=payload.user_id)
				except Exception:
					pass
			return ChatResponse(success=True, data=ChatResponseData(response="memory cleared"))

		# Optional reset of memory for this conversation
		if payload.reset_memory and thread_id:
			try:
				if global_memory_store:
					global_memory_store().clear(chat_id=thread_id, user_id=None)
			except Exception:
				pass

		# Ensure chat pipeline is enabled to leverage LangGraph memory/checkpointer
		if os.getenv("GRAPH_CHAT_ENABLED") is None and os.getenv("GRAPH_ENABLED") is None:
			os.environ["GRAPH_CHAT_ENABLED"] = "true"

		# Heuristic dispatch without explicit modes
		# 1) Chart-like queries → graph chat (it yields charts and responses)
		if _is_chart_intent(text):
			result_graph = run_chat_graph({
				"query": text,
				"user_id": payload.user_id,
				"chat_id": payload.chat_id,
			})
			resp = result_graph.get("response", "")
			return ChatResponse(
				success=True,
				data=ChatResponseData(
					response=resp or "",
					chart_data=result_graph.get("chart_data"),
					charts=result_graph.get("charts"),
				),
			)

		# 2) Finance-like queries → super agent for better company/ticker reasoning
		if _is_finance_intent(text):
			result = run_super_agent(text, memory={}, thread_id=thread_id)
			return ChatResponse(success=True, data=ChatResponseData(response=result.get("output", "")))

		# 3) Default: graph chat
		result_graph = run_chat_graph({
			"query": text,
			"user_id": payload.user_id,
			"chat_id": payload.chat_id,
		})
		resp = result_graph.get("response", "")
		return ChatResponse(
			success=True,
			data=ChatResponseData(
				response=resp or "",
				chart_data=result_graph.get("chart_data"),
				charts=result_graph.get("charts"),
			),
		)
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


class AgentStreamPayload(BaseModel):
	model_config = ConfigDict(extra="ignore")
	question: Optional[str] = None
	query: Optional[str] = None
	user_id: Optional[str] = None
	chat_id: Optional[str] = None


@router.post("/stream")
async def run_agent_stream(payload: AgentStreamPayload):
	try:
		agent = build_agent()
		sys_msg_text = (
			"You can search the web using EXA tools when needed. Prefer up-to-date sources."
		)
		text = (payload.question or payload.query or "").strip()
		if not text:
			raise HTTPException(status_code=400, detail="query or question is required")

		# Build message objects if langchain_core is available, else dicts
		if HumanMessage and SystemMessage:
			system = SystemMessage(content=sys_msg_text)
			user = HumanMessage(content=text)
			messages: list[Any] = [system, user]
		else:
			messages = [
				{"type": "system", "content": sys_msg_text},
				{"type": "human", "content": text},
			]

		async def token_generator():
			index = 0
			sent_any = False
			try:
				# Prefer async streaming interfaces if present on the agent
				if hasattr(agent, "astream_events") and callable(getattr(agent, "astream_events")):
					async for event in agent.astream_events({"messages": messages, "thread_id": payload.chat_id or payload.user_id}, version="v1"):
						ev_name = ""
						data_obj = None
						if isinstance(event, dict):
							ev_name = str(event.get("event") or event.get("type") or "")
							data_obj = event.get("data")
						else:
							ev_name = str(getattr(event, "event", "") or getattr(event, "type", ""))
							data_obj = getattr(event, "data", None) or event
						text = ""
						if ev_name in ("on_chat_model_stream", "on_llm_stream"):
							if isinstance(data_obj, dict):
								possible_chunk = data_obj.get("chunk") or data_obj.get("token") or ""
								if hasattr(possible_chunk, "content"):
									text = getattr(possible_chunk, "content", None) or ""
								elif hasattr(possible_chunk, "delta"):
									text = getattr(possible_chunk, "delta", None) or ""
								elif isinstance(possible_chunk, str):
									text = possible_chunk
							else:
								text = getattr(data_obj, "content", None) or getattr(data_obj, "delta", None) or ""
						if text:
							sent_any = True
							yield (json.dumps({"text": text, "end": False, "index": index}) + "\n").encode("utf-8")
							index += 1
				elif hasattr(agent, "astream") and callable(getattr(agent, "astream")):
					async for event in agent.astream({"messages": messages, "thread_id": payload.chat_id or payload.user_id}, stream_mode="values"):
						msgs = event.get("messages") if isinstance(event, dict) else None
						if not msgs:
							continue
						last = msgs[-1]
						last_type = getattr(last, "type", None) or (last.get("type") if isinstance(last, dict) else None)
						if last_type != "ai":
							continue
						text = getattr(last, "content", None) or (last.get("content") if isinstance(last, dict) else "")
						if not text:
							continue
						sent_any = True
						chunk_size = 200
						for i in range(0, len(text), chunk_size):
							piece = text[i:i+chunk_size]
							if piece:
								yield (json.dumps({"text": piece, "end": False, "index": index}) + "\n").encode("utf-8")
								index += 1
			except Exception:
				# swallow streaming errors and fall back to invoke
				sent_any = False
			if not sent_any:
				try:
					resp = agent.invoke({"messages": messages, "thread_id": payload.chat_id or payload.user_id})
					content = ""
					if isinstance(resp, dict):
						msgs = resp.get("messages")
						if isinstance(msgs, list) and msgs:
							last = msgs[-1]
							content = getattr(last, "content", None) or (last.get("content") if isinstance(last, dict) else "")
					if content:
						chunk_size = 200
						for i in range(0, len(content), chunk_size):
							piece = content[i:i+chunk_size]
							if piece:
								yield (json.dumps({"text": piece, "end": False, "index": index}) + "\n").encode("utf-8")
								index += 1
				except Exception:
					pass
			# Final end marker
			yield (json.dumps({"end": True}) + "\n").encode("utf-8")

		headers = {
			"Cache-Control": "no-cache, no-transform",
			"X-Accel-Buffering": "no",
		}
		return StreamingResponse(token_generator(), media_type="text/plain", headers=headers)
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


# Frontend compatibility alias: POST /stream/messages → same as /agent/stream
@stream_alias_router.post("/stream/messages")
async def run_stream_messages_alias(payload: AgentStreamPayload):
	return await run_agent_stream(payload)
