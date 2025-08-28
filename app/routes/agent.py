from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from starlette.responses import StreamingResponse
import json

from app.agents.super.runner import run_super_agent

router = APIRouter(prefix="/agent", tags=["agent"])


class AgentPayload(BaseModel):
    question: str


@router.post("")
def run_agent(payload: AgentPayload):
    try:
        result = run_super_agent(payload.question, memory={})
        return {"answer": result.get("output", "")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class AgentStreamPayload(BaseModel):
    question: str
    user_id: str | None = None
    chat_id: str | None = None


@router.post("/stream")
async def run_agent_stream(payload: AgentStreamPayload):
    try:
        agent = build_agent()
        system = SystemMessage(content=(
            "You can search the web using EXA tools when needed. Prefer up-to-date sources."
        ))
        user = HumanMessage(content=payload.question)

        async def token_generator():
            index = 0
            sent_any = False
            try:
                if hasattr(agent, "astream_events"):
                    async for event in agent.astream_events({"messages": [system, user]}, version="v1"):
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
                else:
                    async for event in agent.astream({"messages": [system, user]}, stream_mode="values"):
                        msgs = event.get("messages") if isinstance(event, dict) else None
                        if not msgs:
                            continue
                        last = msgs[-1]
                        if getattr(last, "type", "") != "ai":
                            continue
                        text = getattr(last, "content", None) or ""
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
                sent_any = False
            if not sent_any:
                try:
                    resp = agent.invoke({"messages": [system, user]})
                    content = getattr(resp.get("messages", [])[-1], "content", "") if isinstance(resp, dict) and resp.get("messages") else ""
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
