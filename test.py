import os
import requests
import asyncio
from fastapi import HTTPException
from dotenv import load_dotenv
from openai import AzureOpenAI  # ✅ new client

load_dotenv()  # loads .env automatically

# Exa API
EXA_API_KEY = os.getenv("EXA_API_KEY")

# Azure OpenAI API
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")  # e.g. https://<resource>.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")  # your deployment name

if not EXA_API_KEY:
    raise HTTPException(status_code=500, detail="Exa API key not configured")

# ✅ Initialize Azure OpenAI client globally
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2023-07-01-preview"
)


class ExaClient:
    def __init__(self):
        self.base_url = "https://api.exa.ai/search"  # ✅ fixed endpoint

    def get_news(self, company: str, limit: int = 10):
        if not EXA_API_KEY:
            raise HTTPException(status_code=500, detail="Exa API key not configured")

        headers = {
            "x-api-key": EXA_API_KEY,  # ✅ correct header
            "Content-Type": "application/json"
        }
        payload = {
            "query": f"{company} financial news",
            "type": "auto",     # ✅ valid value: "keyword" or "neural"
            "category": "news"
        }

        try:
            resp = requests.post(self.base_url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

            articles = []
            for item in data.get("results", []):
                articles.append({
                    "title": item.get("title", ""),
                    "content": item.get("text", "") or item.get("snippet", ""),  # fallback to snippet
                    "url": item.get("url", "")
                })
            return articles
        except requests.exceptions.HTTPError as e:
            raise HTTPException(status_code=500, detail=f"Exa API request failed: {str(e)} | Response: {resp.text}")
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Network error while contacting Exa API: {str(e)}")


async def summarize_batch(company: str, articles: list) -> list:
    """
    Summarize all articles in one batch prompt asynchronously.
    Returns a list of summaries in the same order.
    """
    if not articles:
        return []

    if not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT:
        raise HTTPException(status_code=500, detail="Azure OpenAI configuration missing")

    # Build batch prompt
    batch_prompt = f"Summarize the following financial news articles about {company}.\n"
    for idx, article in enumerate(articles, start=1):
        text = article["content"] or article["title"]
        batch_prompt += f"\nArticle {idx}:\n{text}\nSource link: {article['url']}\n"

    batch_prompt += "\nInstructions: Summarize each article in 3-4 concise sentences. Keep summaries factual, focus on company performance, stock impact, and financial news. Include the source link at the end of each summary."

    # ✅ Call Azure OpenAI in one request using new client
    try:
        resp = await asyncio.to_thread(
            client.chat.completions.create,
            model=AZURE_OPENAI_DEPLOYMENT,   # deployment name
            messages=[{"role": "user", "content": batch_prompt}],
            temperature=0.2,
            max_tokens=1500
        )
        summary_text = resp.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Azure OpenAI request failed: {str(e)}")

    # Split summaries by article (assuming model keeps numbering)
    summaries = summary_text.split("\n\n")
    if len(summaries) != len(articles):
        summaries = [summary_text] * len(articles)

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

    return {
        "company": company,
        "insights": insights,
        "all_sources": all_sources
    }
