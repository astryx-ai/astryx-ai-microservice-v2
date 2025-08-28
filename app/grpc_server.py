import os
import sys
import asyncio
from typing import Optional, Dict, Any
import grpc  # type: ignore
from grpc_reflection.v1alpha import reflection

# Ensure generated gRPC modules are importable as top-level (message_pb2, message_pb2_grpc)
PROTO_DIR = os.path.join(os.path.dirname(__file__), "proto")
if PROTO_DIR not in sys.path:
    sys.path.insert(0, PROTO_DIR)

# Lazy import to avoid hard fail if protos not generated yet
try:
    import message_pb2  # type: ignore
    import message_pb2_grpc  # type: ignore
except Exception:  # pragma: no cover
    message_pb2 = None  # type: ignore
    message_pb2_grpc = None  # type: ignore

# Delegate business logic to handlers
from app.interfaces.grpc.handlers import message_stream_chunks, handle_get_chart
from app.graph.runner import run_chart_async

GRPC_PORT = int(os.getenv("GRPC_PORT", "50051"))

class MessageService(message_pb2_grpc.MessageServiceServicer):  # type: ignore[attr-defined]
    async def MessageStream(self, request, context):  # type: ignore[override]
        async for chunk in message_stream_chunks(message_pb2, request):
            await context.write(chunk)
        return

    async def GetChart(self, request, context):  # type: ignore[override]
        use_graph = os.getenv("GRAPH_ENABLED", "").lower() in ("1","true","yes","on") or os.getenv("GRAPH_CHART_ENABLED", "").lower() in ("1","true","yes","on")
        if use_graph:
            inputs = {
                "query": getattr(request, "query", ""),
                "symbol": getattr(request, "symbol", ""),
                "chart_type": getattr(request, "chart_type", ""),
                "range": getattr(request, "range", ""),
                "interval": getattr(request, "interval", ""),
                "title": getattr(request, "title", ""),
                "description": getattr(request, "description", ""),
                "user_id": getattr(request, "user_id", ""),
                "chat_id": getattr(request, "chat_id", ""),
            }
            result = await run_chart_async(inputs)
            return message_pb2.ChartResponse(json=result.get("json", "{}"), content_type=result.get("content_type", "application/json"))
        # Legacy behavior (default)
        resp = await handle_get_chart(message_pb2, request)
        return resp

async def start_grpc_server(port: Optional[int] = None):
    server = grpc.aio.server()  # type: ignore[attr-defined]
    if message_pb2_grpc is not None and message_pb2 is not None:
        message_pb2_grpc.add_MessageServiceServicer_to_server(MessageService(), server)  # type: ignore[attr-defined]
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


