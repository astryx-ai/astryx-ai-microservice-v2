import asyncio
import os
import sys
from typing import Optional

import grpc

# Ensure generated modules in app/proto are importable as top-level for grpc imports
_PROTO_DIR = os.path.join(os.path.dirname(__file__), "proto")
if _PROTO_DIR not in sys.path:
    sys.path.insert(0, _PROTO_DIR)

import test_pb2  # type: ignore
import test_pb2_grpc  # type: ignore


GRPC_PORT = int(os.getenv("GRPC_PORT", "50051"))


class TestService(test_pb2_grpc.TestServiceServicer):
    async def Ping(self, request: test_pb2.PingRequest, context: grpc.aio.ServicerContext) -> test_pb2.PingReply:  # type: ignore[name-defined]
        msg = request.message or ""
        return test_pb2.PingReply(message=f"pong: {msg}")


async def start_grpc_server(port: int = GRPC_PORT) -> grpc.aio.Server:  # type: ignore[name-defined]
    server: grpc.aio.Server = grpc.aio.server()  # type: ignore[name-defined]
    test_pb2_grpc.add_TestServiceServicer_to_server(TestService(), server)
    server.add_insecure_port(f"[::]:{port}")
    await server.start()
    return server


async def stop_grpc_server(server: Optional[grpc.aio.Server]) -> None:  # type: ignore[name-defined]
    if server is None:
        return
    # Give ongoing RPCs a brief grace period
    await server.stop(grace=2.0)


async def serve_forever() -> None:
    server = await start_grpc_server(GRPC_PORT)
    try:
        await server.wait_for_termination()
    except (KeyboardInterrupt, asyncio.CancelledError):
        await stop_grpc_server(server)


if __name__ == "__main__":
    asyncio.run(serve_forever())


