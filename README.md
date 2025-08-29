# Astryx AI Microservice

FastAPI microservice with Azure OpenAI chat endpoint.

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

## Next Steps / Ideas

- Add proper cost calculation based on Azure pricing.
- Persist conversation history per chat_id.
- Implement streaming responses (Server-Sent Events or WebSocket).
- Add unit tests (pytest) for service & routes.
