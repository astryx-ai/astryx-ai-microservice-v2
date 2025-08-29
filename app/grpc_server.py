import asyncio
import os
import sys
from typing import Optional, Tuple

import grpc

from langchain_core.messages import HumanMessage, SystemMessage
from app.services.agent import build_agent
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

    class MessageService(message_pb2_grpc.MessageServiceServicer):  # type: ignore[attr-defined]
        async def MessageStream(self, request, context):  # type: ignore[override]
            print(f"[gRPC] MessageStream received: user_id={getattr(request,'user_id',None)}, chat_id={getattr(request,'chat_id',None)}, query={getattr(request,'query',None)!r}")
            agent = build_agent()
            system = SystemMessage(content=(
                "You can search the web using EXA tools when needed. Prefer up-to-date sources."
            ))
            user = HumanMessage(content=request.query)
            index = 0
            sent_any = False
            try:
                # Prefer fine-grained token streaming if available
                if hasattr(agent, "astream_events"):
                    async for event in agent.astream_events({"messages": [system, user]}, version="v1"):
                        ev_name = ""
                        data_obj = None
                        # Normalize event and data across lc versions
                        if isinstance(event, dict):
                            ev_name = str(event.get("event") or event.get("type") or "")
                            data_obj = event.get("data")
                        else:
                            ev_name = str(getattr(event, "event", "") or getattr(event, "type", ""))
                            data_obj = getattr(event, "data", None) or event

                        # Extract text from known chunk shapes
                        text = ""
                        if ev_name == "on_chat_model_stream" or ev_name == "on_llm_stream":
                            # data_obj can be a dict or an AIMessageChunk-like object
                            if isinstance(data_obj, dict):
                                possible_chunk = data_obj.get("chunk") or data_obj.get("token") or ""
                                if hasattr(possible_chunk, "content"):
                                    text = getattr(possible_chunk, "content", None) or ""
                                elif hasattr(possible_chunk, "delta"):
                                    text = getattr(possible_chunk, "delta", None) or ""
                                elif isinstance(possible_chunk, str):
                                    text = possible_chunk
                            else:
                                # AIMessageChunk-like object
                                text = getattr(data_obj, "content", None) or getattr(data_obj, "delta", None) or ""

                        if text:
                            print(f"[gRPC] stream chunk: {text[:60]!r}")
                            sent_any = True
                            yield message_pb2.MessageChunk(text=text, end=False, index=index)  # type: ignore[attr-defined]
                            index += 1
                        # Ignore other events; gRPC stream close will signal end
                else:
                    # Fallback to message-level streaming (larger chunks)
                    async for event in agent.astream({"messages": [system, user]}, stream_mode="values"):
                        msgs = event.get("messages") if isinstance(event, dict) else None
                        if not msgs:
                            continue
                        last = msgs[-1]
                        last_type = getattr(last, "type", "")
                        if last_type != "ai":
                            continue
                        text = getattr(last, "content", None) or ""
                        if not text:
                            continue
                        print(f"[gRPC] stream chunk: {text[:60]!r}")
                        sent_any = True
                        # Re-chunk larger message-level emissions to encourage progressive delivery
                        chunk_size = 200
                        for i in range(0, len(text), chunk_size):
                            piece = text[i:i+chunk_size]
                            if piece:
                                yield message_pb2.MessageChunk(text=piece, end=False, index=index)  # type: ignore[attr-defined]
                                index += 1
            except Exception as e:
                print(f"[gRPC] agent streaming error: {e}")
                sent_any = False

            if not sent_any:
                # Final fallback: single-shot invoke and manual chunking
                try:
                    resp = agent.invoke({"messages": [system, user]})
                    content = getattr(resp.get("messages", [])[-1], "content", "") if isinstance(resp, dict) and resp.get("messages") else ""
                    if content:
                        chunk_size = 200
                        for i in range(0, len(content), chunk_size):
                            piece = content[i:i+chunk_size]
                            if piece:
                                yield message_pb2.MessageChunk(text=piece, end=False, index=index)  # type: ignore[attr-defined]
                                index += 1
                except Exception:
                    pass

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


