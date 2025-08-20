# Astryx AI Microservice

FastAPI microservice with Azure OpenAI chat endpoint.

## Project Structure

```text
app/
	config.py          # Settings & .env loading via pydantic BaseSettings
	schemas.py         # Pydantic request/response models
	routes/
		chat.py          # /chat/message endpoint
	services/
		azure_openai.py  # Azure OpenAI service wrapper
main.py              # App factory & router include
.env.example         # Example environment variables
```

## Environment Variables (.env)

Copy `.env.example` to `.env` and fill in values:

```env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_DEPLOYMENT=your_deployment_name
AZURE_OPENAI_API_VERSION=2024-05-01-preview
ENVIRONMENT=local
```

## Install & Run (uvicorn)

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

## RAG + PGVector extension

New endpoints:
- POST `/scrape/{symbol}`: Scrape stock fundamentals + news; chunk+embed into PGVector.
- POST `/ask`: Body `{ "query": "...", "symbol": "TCS", "user_id": "u1", "chat_id": "c1" }` -> LLM answer with retrieved context.
- GET `/candles/{symbol}?days=30&stream=false`: OHLCV via yfinance; `stream=true` for SSE.

### Configure
Copy `.env.example` to `.env` and set at least:
- `OPENAI_API_KEY`
- `DATABASE_URL` (Postgres with pgvector installed; extension auto-create attempted)

### Install & run

1. Install deps
2. Create DB tables
3. Start server

Optional: on first run, call `/scrape/{symbol}` before asking to ensure vector store has content.

## RAG + PGVector Additions

This service now includes:

- Moneycontrol scrapers for stocks and news (already present)
- LangChain + PGVector vector store using OpenAIEmbeddings
- Async SQLAlchemy with asyncpg
- Background refresh of symbols every REFRESH_MINUTES
- Endpoints:
  - `POST /scrape/{symbol}` — scrape and (re)index
  - `POST /ask` — RAG question answering over stored vectors
  - `GET /candles/{symbol}?days=30&stream=false` — OHLCV from yfinance; set `stream=true` for SSE

### Configure

Copy `.env.example` to `.env` and set:

- `DATABASE_URL` (Postgres with PGVector extension)
- `OPENAI_API_KEY`

### Install

Use your env manager; then install requirements.

### Initialize DB

Run the helper to create tables:

python -m app.migrations_helper

### Run

uvicorn main:app --reload

```

## Test Root Endpoint

`GET http://localhost:8000/`

## Chat Endpoint

`POST http://localhost:8000/chat/message`

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

## Next Steps / Ideas

- Add proper cost calculation based on Azure pricing.
- Persist conversation history per chat_id.
- Implement streaming responses (Server-Sent Events or WebSocket).
- Add unit tests (pytest) for service & routes.
