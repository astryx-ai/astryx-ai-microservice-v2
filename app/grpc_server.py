import asyncio
import os
import sys
from typing import Optional, Tuple, Dict, Any

import grpc

from app.services.super_agent import run_super_agent
from grpc_reflection.v1alpha import reflection


GRPC_PORT = int(os.getenv("GRPC_PORT", "50051"))


def _ensure_proto_path() -> str:
    proto_dir = os.path.join(os.path.dirname(__file__), "proto")
    if proto_dir not in sys.path:
        sys.path.insert(0, proto_dir)
    return proto_dir


def _ensure_message_stubs() -> Tuple[object, object]:
    _ensure_proto_path()
    try:
        import message_pb2  # type: ignore
        import message_pb2_grpc  # type: ignore
        return message_pb2, message_pb2_grpc
    except Exception:
        # Attempt runtime generation of stubs
        try:
            from grpc_tools import protoc  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"grpc_tools.protoc unavailable: {e}")
        proto_dir = _ensure_proto_path()
        proto_file = os.path.join(proto_dir, "message.proto")
        args = [
            "protoc",
            f"-I{proto_dir}",
            f"--python_out={proto_dir}",
            f"--grpc_python_out={proto_dir}",
            proto_file,
        ]
        code = protoc.main(args)
        if code != 0:  # pragma: no cover
            raise RuntimeError(f"protoc failed with exit code {code}")
        import importlib
        message_pb2 = importlib.import_module("message_pb2")  # type: ignore
        message_pb2_grpc = importlib.import_module("message_pb2_grpc")  # type: ignore
        return message_pb2, message_pb2_grpc


async def start_grpc_server(port: int = GRPC_PORT) -> grpc.aio.Server:  # type: ignore[name-defined]
    server: grpc.aio.Server = grpc.aio.server()  # type: ignore[name-defined]

    message_pb2, message_pb2_grpc = _ensure_message_stubs()

    # Simple in-memory store keyed by chat_id or user_id
    _MEM: Dict[str, Dict[str, Any]] = {}

    class MessageService(message_pb2_grpc.MessageServiceServicer):  # type: ignore[attr-defined]
        async def MessageStream(self, request, context):  # type: ignore[override]
            print(f"[gRPC] MessageStream received: user_id={getattr(request,'user_id',None)}, chat_id={getattr(request,'chat_id',None)}, query={getattr(request,'query',None)!r}")
            # Use SuperAgent (Yahoo-based) and stream chunked final output
            key = getattr(request, "chat_id", None) or getattr(request, "user_id", None)
            mem = _MEM.get(key, {}) if key else {}
            try:
                # Run sync SuperAgent without blocking the event loop
                result = await asyncio.to_thread(run_super_agent, request.query, mem)
                if key:
                    _MEM[key] = mem
                content = (result or {}).get("output") or ""
            except Exception as e:
                print(f"[gRPC] super_agent error: {e}")
                content = ""

            index = 0
            if content:
                # Emit fast, word-level chunks for snappier UX
                words = content.split()
                if len(words) > 1:
                    for w in words:
                        text = (w + " ")
                        print(f"[gRPC] stream chunk: {text[:60]!r}")
                        yield message_pb2.MessageChunk(text=text, end=False, index=index)  # type: ignore[attr-defined]
                        index += 1
                else:
                    # Fallback to small fixed-size chunks
                    chunk_size = 32
                    for i in range(0, len(content), chunk_size):
                        piece = content[i:i+chunk_size]
                        if piece:
                            print(f"[gRPC] stream chunk: {piece[:60]!r}")
                            yield message_pb2.MessageChunk(text=piece, end=False, index=index)  # type: ignore[attr-defined]
                            index += 1

    message_pb2_grpc.add_MessageServiceServicer_to_server(MessageService(), server)  # type: ignore[attr-defined]

    # Enable server reflection for tooling like grpcurl
    SERVICE_NAMES = (
        message_pb2.DESCRIPTOR.services_by_name['MessageService'].full_name,  # type: ignore[attr-defined]
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)
    server.add_insecure_port(f"[::]:{port}")
    await server.start()
    return server


async def stop_grpc_server(server: Optional[grpc.aio.Server]) -> None:  # type: ignore[name-defined]
    if server is None:
        return
    await server.stop(grace=2.0)


async def serve_forever() -> None:
    server = await start_grpc_server(GRPC_PORT)
    try:
        await server.wait_for_termination()
    except (KeyboardInterrupt, asyncio.CancelledError):
        await stop_grpc_server(server)


if __name__ == "__main__":
    asyncio.run(serve_forever())


