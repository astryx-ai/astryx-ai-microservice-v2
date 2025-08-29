from fastapi import FastAPI
from app.routes.health import router as health_router
from app.routes.ingest_news import router as ingest_news_router
from app.routes.chat import router as chat_router
from app.routes.ingest_companies import router as ingest_companies_router
from app.routes.agent import router as agent_router

app = FastAPI(title="Astryx AI Microservice")

# gRPC server lifecycle
@app.on_event("startup")
async def _start_grpc():
    try:
        from app.grpc_server import start_grpc_server
        app.state.grpc_server = await start_grpc_server()
    except Exception as e:
        # Non-fatal: allow HTTP server to continue even if gRPC failed
        print(f"[startup] gRPC failed to start: {e}")


@app.on_event("shutdown")
async def _stop_grpc():
    try:
        server = getattr(app.state, "grpc_server", None)
        if server:
            from app.grpc_server import stop_grpc_server
            await stop_grpc_server(server)
    except Exception as e:
        print(f"[shutdown] gRPC failed to stop: {e}")

# Health stays root
app.include_router(health_router)

# Add prefixes
app.include_router(chat_router)
app.include_router(agent_router)
app.include_router(ingest_news_router)
app.include_router(ingest_companies_router)
