# Astryx AI Microservice

FastAPI microservice with Azure OpenAI chat endpoint.

## Canonical API Endpoints

- POST `http://localhost:8000/chat` — Unified Chat API
  - Chat with optional charts in response (default mode)
  - Super agent: set `{ "mode": "super" }` and optionally `{"format": "md"}` for Markdown
  - Lightweight agent: set `{ "mode": "agent" }`
  - Clear memory: set `{ "memory_clear": true, "chat_id"|"user_id": "..." }`
  - Deprecated alias: POST `/chat/message` (30-day sunset from 2025-08-29). Logs a warning and forwards to `/chat`.
- POST `http://localhost:8000/ingest/news` — Ingest news
- POST `http://localhost:8000/ingest/companies` — Ingest companies
- GET  `http://localhost:8000/health` — Liveness

Deprecated shims (forward to `/chat` and log warnings):

- `POST /agent` — use `/chat` with `{ "mode": "agent" }`
- `POST /super/chat` — use `/chat` with `{ "mode": "super" }`
- `POST /super/memory/clear` — use `/chat` with `{ "memory_clear": true }`

## Chat Endpoint

`POST http://localhost:8000/chat`

Request body:

```json
{
  "query": "Hello there",
  "user_id": "u1",
  "chat_id": "c1"
}
```

Response body (example):

```json
{
  "success": true,
  "data": {
    "response": "Hi! How can I help you today?",
    "chart_data": null,
    "tokens_used": 42,
    "cost": 0.0
  }
}
```

Deprecated compatibility endpoint:

- `POST /chat/message` — forwards to `/chat` and logs a deprecation warning. Will be removed after the 30-day sunset.

## Next Steps / Ideas

- Add proper cost calculation based on Azure pricing.
- Persist conversation history per chat_id.
- Implement streaming responses (Server-Sent Events or WebSocket).
- Add unit tests (pytest) for service & routes.
