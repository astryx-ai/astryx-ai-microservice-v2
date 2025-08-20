from fastapi import APIRouter, HTTPException
from app.schemas import AIChatRequest, AIChatResponse, AIChatResponseData
from app.services.azure_openai import azure_openai_service

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/message", response_model=AIChatResponse)
async def chat_message(req: AIChatRequest):
    if not azure_openai_service.is_configured():
        return AIChatResponse(success=False, error="Azure OpenAI not configured. Set env vars in .env")
    try:
        response, tokens_used = await azure_openai_service.chat(req.query)
        return AIChatResponse(
            success=True,
            data=AIChatResponseData(
                response=response,
                chart_data=None,
                tokens_used=tokens_used,
                cost=0.0,  # placeholder
            ),
        )
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(e))
