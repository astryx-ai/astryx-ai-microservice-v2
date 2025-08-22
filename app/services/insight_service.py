import os
import requests
import asyncio
import json
from fastapi import HTTPException
from dotenv import load_dotenv
from openai import AzureOpenAI
from langchain.schema import Document
from app.scrapper.sanitize import clean_text
from app.services.rag import chunk_text, upsert_news
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Exa API
EXA_API_KEY = os.getenv("EXA_API_KEY")

# Azure OpenAI API
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

if not EXA_API_KEY:
    raise HTTPException(status_code=500, detail="Exa API key not configured")

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2023-07-01-preview"
)

class ExaClient:
    def __init__(self):
        self.base_url = "https://api.exa.ai/search"

    def get_news(self, company: str, limit: int = 3):
        if not EXA_API_KEY:
            raise HTTPException(status_code=500, detail="Exa API key not configured")

        headers = {
            "x-api-key": EXA_API_KEY,
            "Content-Type": "application/json"
        }
        payload = {
            "query": f"{company} financial news",
            "type": "auto",
            "category": "news"
        }

        try:
            logger.info(f"Sending request to Exa API: {self.base_url} with payload: {payload}")
            resp = requests.post(self.base_url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            logger.info(f"Exa API response: {data}")

            articles = []
            for item in data.get("results", []):
                articles.append({
                    "title": item.get("title", ""),
                    "content": item.get("text", "") or item.get("snippet", ""),
                    "url": item.get("url", "")
                })
            return articles[:limit]
        except requests.exceptions.HTTPError as e:
            logger.error(f"Exa API request failed: {str(e)}, Response: {resp.text}")
            raise HTTPException(status_code=500, detail=f"Exa API request failed: {str(e)}, Response: {resp.text}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error while contacting Exa API: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Network error while contacting Exa API: {str(e)}")

async def summarize_batch(company: str, articles: list) -> list:
    if not articles:
        return []

    if not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT:
        raise HTTPException(status_code=500, detail="Azure OpenAI configuration missing")

    batch_prompt = f"Summarize the following financial news articles about {company}. Separate each article summary with exactly '---' on a new line.\n"
    for idx, article in enumerate(articles, start=1):
        text = article["content"] or article["title"]
        batch_prompt += f"\nArticle {idx}:\n{text}\nSource link: {article['url']}\n"

    batch_prompt += "\nInstructions: Summarize each article in 3-4 concise sentences. Keep summaries factual, focus on company performance, stock impact, and financial news. Include the source link at the end of each summary. Separate each summary with exactly '---' on a new line."

    try:
        logger.info("Sending request to Azure OpenAI")
        resp = await asyncio.to_thread(
            client.chat.completions.create,
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[{"role": "user", "content": batch_prompt}],
            temperature=0.2,
            max_tokens=1200
        )
        summary_text = resp.choices[0].message.content.strip()
        logger.info(f"Azure OpenAI response: {summary_text}")
    except Exception as e:
        logger.error(f"Azure OpenAI request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Azure OpenAI request failed: {str(e)}")

    summaries = [s.strip() for s in summary_text.split("---") if s.strip()]
    logger.info(f"Split summaries: {summaries}")
    
    if len(summaries) != len(articles):
        logger.warning(f"Summary splitting failed: expected {len(articles)} summaries, got {len(summaries)}. Returning empty summaries.")
        return ["No summary available due to response parsing error"] * len(articles)

    return summaries

async def get_company_insights(company: str, limit: int = 10):
    exa = ExaClient()
    try:
        articles = exa.get_news(company, limit=limit)
    except HTTPException as e:
        raise e

    summaries = await summarize_batch(company, articles)

    insights = []
    all_sources = []

    for article, summary in zip(articles, summaries):
        insights.append({
            "summary": summary,
            "source": article["url"]
        })
        all_sources.append(article["url"])

    # Prepare JSON response
    result = {
        "company": company,
        "insights": insights,
        "all_sources": all_sources
    }

    # Store the entire JSON response in vector database
    async def store_json_response():
        try:
            # Convert JSON to string for storage
            json_content = json.dumps(result)
            blob = clean_text(json_content)
            if not blob:
                logger.warning("Cleaned JSON content is empty, skipping storage")
                return 0
            meta: Dict[str, Any] = {
                "company": company,
                "type": "insight",
                "source": "exa_summary"
            }
            docs = chunk_text(blob, meta)
            if docs:
                await asyncio.to_thread(upsert_news, docs)
                logger.info(f"Stored {len(docs)} document chunks in vector DB")
            return len(docs)
        except Exception as e:
            logger.error(f"Failed to store JSON response in vector DB: {str(e)}")
            return 0

    # Start storage task concurrently
    storage_task = asyncio.create_task(store_json_response())

    # Await storage task
    ingested_count = await storage_task

    return {
        "company": company,
        "insights": insights,
        "all_sources": all_sources,
        "ingested": ingested_count  # Include for logging/debugging
    }