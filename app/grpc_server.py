import asyncio
import os
import sys
import json
import re
from typing import Optional, Dict, Any

import grpc
from grpc_reflection.v1alpha import reflection

# Ensure generated gRPC modules are importable as top-level (message_pb2, message_pb2_grpc)
PROTO_DIR = os.path.join(os.path.dirname(__file__), "proto")
if PROTO_DIR not in sys.path:
    sys.path.insert(0, PROTO_DIR)

import message_pb2  # type: ignore
import message_pb2_grpc  # type: ignore

from app.services.super_agent import run_super_agent
from app.services.charts.builder import build_chart  # type: ignore


GRPC_PORT = int(os.getenv("GRPC_PORT", "50051"))
CHART_CHUNK_SIZE = 4096
_MEM: Dict[str, Dict[str, Any]] = {}


class MessageService(message_pb2_grpc.MessageServiceServicer):  # type: ignore
    async def MessageStream(self, request, context):  # type: ignore[override]
        print(
            f"[gRPC] MessageStream received: user_id={getattr(request,'user_id',None)}, "
            f"chat_id={getattr(request,'chat_id',None)}, query={getattr(request,'query',None)!r}"
        )

        key = getattr(request, "chat_id", None) or getattr(request, "user_id", None)
        mem = _MEM.get(key, {}) if key else {}

        try:
            q = getattr(request, "query", "") or ""
            # Detect chart queries and short-circuit with chart envelope
            if re.search(r"\b(candle|candlestick|ohlc|chart)\b", q, re.I):
                payload = await build_chart(query=q)
                envelope = {"kind": "chart", "payload": payload}
                text = json.dumps(envelope)
                idx = 0
                for i in range(0, len(text), CHART_CHUNK_SIZE):
                    piece = text[i:i + CHART_CHUNK_SIZE]
                    if piece:
                        yield message_pb2.MessageChunk(text=piece, end=False, index=idx)  # type: ignore[attr-defined]
                        idx += 1
                return

            # Fallback: run super agent and stream the response text in small chunks
            result = await asyncio.to_thread(run_super_agent, request.query, mem)
            if key:
                _MEM[key] = mem
            content = (result or {}).get("output") or ""
        except Exception as e:
            print(f"[gRPC] super_agent error: {e}")
            content = ""

        index = 0
        if content:
            words = content.split()
            if len(words) > 1:
                for w in words:
                    text = (w + " ")
                    yield message_pb2.MessageChunk(text=text, end=False, index=index)  # type: ignore[attr-defined]
                    index += 1
            else:
                chunk_size = 32
                for i in range(0, len(content), chunk_size):
                    piece = content[i:i + chunk_size]
                    if piece:
                        yield message_pb2.MessageChunk(text=piece, end=False, index=index)  # type: ignore[attr-defined]
                        index += 1

    async def GetChart(self, request, context):  # type: ignore[override]
        q = (getattr(request, "query", "") or "").strip()
        symbol = (getattr(request, "symbol", "") or "").strip()
        chart_type = (getattr(request, "chart_type", "") or "").strip().lower()
        range_ = (getattr(request, "range", "") or "").strip() or "1d"
        interval = (getattr(request, "interval", "") or "").strip() or "5m"
        title = getattr(request, "title", "") or ""
        description = getattr(request, "description", "") or ""

        payload = await build_chart(
            query=q,
            symbol=symbol,
            chart_type=chart_type,
            range_=range_,
            interval=interval,
            title=title,
            description=description,
        )
        return message_pb2.ChartResponse(json=json.dumps(payload), content_type="application/json")  # type: ignore[attr-defined]


async def start_grpc_server(port: Optional[int] = None):
    server = grpc.aio.server()  # type: ignore[attr-defined]
    message_pb2_grpc.add_MessageServiceServicer_to_server(MessageService(), server)  # type: ignore[attr-defined]

    # Enable reflection for tooling (e.g., grpcurl)
    service_names = (
        message_pb2.DESCRIPTOR.services_by_name["MessageService"].full_name,  # type: ignore[attr-defined]
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(service_names, server)

    use_port = int(port or GRPC_PORT)
    server.add_insecure_port(f"[::]:{use_port}")
    await server.start()
    print(f"[gRPC] Server started on port {use_port}")
    return server


async def stop_grpc_server(server: Optional[grpc.aio.Server]) -> None:  # type: ignore[name-defined]
    if not server:
        return
    await server.stop(grace=2.0)
    print("[gRPC] Server stopped")


async def serve_forever() -> None:
    server = await start_grpc_server()
    try:
        await server.wait_for_termination()
    except (KeyboardInterrupt, asyncio.CancelledError):
        await stop_grpc_server(server)


if __name__ == "__main__":
    asyncio.run(serve_forever())


