"""
News API Routes
Provides endpoints for fetching company news articles in the exact format needed by frontend.
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging

from app.services.news_service import news_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/news", tags=["news"])


# Response Models (matching Figma design structure)
class NewsArticle(BaseModel):
    """Individual news article model."""
    title: str = Field(..., description="Article headline")
    description: str = Field(..., description="Brief article description/summary")
    url: str = Field(..., description="Article URL")
    publisher: str = Field(..., description="Publisher domain/name")
    published_at: str = Field(..., description="Publication date (YYYY-MM-DD)")
    date_display: str = Field(..., description="Formatted date for display (e.g., 'May 25')")


class CompanyInfo(BaseModel):
    """Company information model."""
    id: str = Field(..., description="Company ID")
    name: str = Field(..., description="Company name")
    nse_symbol: Optional[str] = Field(None, description="NSE trading symbol")
    bse_code: Optional[str] = Field(None, description="BSE scrip code")
    industry: Optional[str] = Field(None, description="Industry sector")
    market_cap: Optional[float] = Field(None, description="Market capitalization")


class CompanyNewsResponse(BaseModel):
    """Response model for company news articles."""
    success: bool = Field(..., description="Request success status")
    company: Optional[CompanyInfo] = Field(None, description="Company information")
    articles: List[NewsArticle] = Field(default_factory=list, description="List of news articles")
    total_articles: int = Field(0, description="Total number of articles returned")
    last_updated: str = Field(..., description="Last update timestamp")
    error: Optional[str] = Field(None, description="Error message if any")


class TrendingNewsResponse(BaseModel):
    """Response model for trending news articles."""
    success: bool = Field(..., description="Request success status")
    articles: List[NewsArticle] = Field(default_factory=list, description="List of trending articles")
    total_articles: int = Field(0, description="Total number of articles returned")
    last_updated: str = Field(..., description="Last update timestamp")
    error: Optional[str] = Field(None, description="Error message if any")


# API Endpoints
@router.get("/company/{company_identifier}", response_model=CompanyNewsResponse)
async def get_company_news(
    company_identifier: str,
    days_back: int = Query(30, ge=1, le=90, description="Number of days to look back for news"),
    max_articles: int = Query(10, ge=1, le=25, description="Maximum number of articles to return")
):
    """
    Get news articles for a specific company.
    
    The company_identifier can be:
    - NSE symbol (e.g., "RELIANCE", "TCS")
    - BSE code (e.g., "500325")
    - Company name (e.g., "Reliance Industries", "Tata Consultancy Services")
    - Partial company name for fuzzy matching
    
    Returns articles in the exact format needed by the frontend News Articles section.
    """
    try:
        logger.info(f"Fetching news for company: {company_identifier}")
        
        result = news_service.fetch_company_news(
            company_identifier=company_identifier,
            days_back=days_back,
            max_articles=max_articles
        )
        
        if not result['success']:
            raise HTTPException(status_code=404, detail=result.get('error', 'Company not found'))
        
        # Convert to response model
        company_info = None
        if result.get('company'):
            company_data = result['company']
            company_info = CompanyInfo(
                id=company_data['id'],
                name=company_data['name'],
                nse_symbol=company_data.get('nse_symbol'),
                bse_code=company_data.get('bse_code'),
                industry=company_data.get('industry'),
                market_cap=company_data.get('market_cap')
            )
        
        articles = [
            NewsArticle(
                title=article['title'],
                description=article['description'],
                url=article['url'],
                publisher=article['publisher'],
                published_at=article['published_at'],
                date_display=article['date_display']
            )
            for article in result.get('articles', [])
        ]
        
        return CompanyNewsResponse(
            success=True,
            company=company_info,
            articles=articles,
            total_articles=result.get('total_articles', len(articles)),
            last_updated=result.get('last_updated', '')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching company news: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/trending", response_model=TrendingNewsResponse)
async def get_trending_news(
    limit: int = Query(15, ge=1, le=30, description="Maximum number of trending articles")
):
    """
    Get trending financial and market news articles.
    
    Returns general market news, financial updates, and business headlines
    in the format needed by the frontend.
    """
    try:
        logger.info("Fetching trending financial news")
        
        result = news_service.get_trending_news(limit=limit)
        
        if not result['success']:
            raise HTTPException(status_code=500, detail=result.get('error', 'Failed to fetch trending news'))
        
        articles = [
            NewsArticle(
                title=article['title'],
                description=article['description'],
                url=article['url'],
                publisher=article['publisher'],
                published_at=article['published_at'],
                date_display=article['date_display']
            )
            for article in result.get('articles', [])
        ]
        
        return TrendingNewsResponse(
            success=True,
            articles=articles,
            total_articles=result.get('total_articles', len(articles)),
            last_updated=result.get('last_updated', '')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching trending news: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/search", response_model=CompanyNewsResponse)
async def search_company_news(
    query: str = Query(..., min_length=2, description="Company search query"),
    max_articles: int = Query(10, ge=1, le=25, description="Maximum number of articles to return")
):
    """
    Search for company news by query.
    
    This endpoint allows flexible searching - the query can be a partial company name,
    ticker symbol, or any search term. It will try to find the best matching company
    and return its news articles.
    """
    try:
        logger.info(f"Searching company news for query: {query}")
        
        result = news_service.fetch_company_news(
            company_identifier=query,
            max_articles=max_articles
        )
        
        if not result['success']:
            # If no exact company match, return empty results rather than error
            return CompanyNewsResponse(
                success=True,
                company=None,
                articles=[],
                total_articles=0,
                last_updated="",
                error=f"No company found matching '{query}'"
            )
        
        # Convert to response model (same as get_company_news)
        company_info = None
        if result.get('company'):
            company_data = result['company']
            company_info = CompanyInfo(
                id=company_data['id'],
                name=company_data['name'],
                nse_symbol=company_data.get('nse_symbol'),
                bse_code=company_data.get('bse_code'),
                industry=company_data.get('industry'),
                market_cap=company_data.get('market_cap')
            )
        
        articles = [
            NewsArticle(
                title=article['title'],
                description=article['description'],
                url=article['url'],
                publisher=article['publisher'],
                published_at=article['published_at'],
                date_display=article['date_display']
            )
            for article in result.get('articles', [])
        ]
        
        return CompanyNewsResponse(
            success=True,
            company=company_info,
            articles=articles,
            total_articles=result.get('total_articles', len(articles)),
            last_updated=result.get('last_updated', '')
        )
        
    except Exception as e:
        logger.error(f"Error searching company news: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/companies/search")
async def search_companies(
    query: str = Query(..., min_length=2, description="Company search query"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of companies to return")
):
    """
    Search for companies by name, symbol, or partial match.
    
    This endpoint helps the frontend implement company search/autocomplete functionality.
    Returns basic company information without news articles.
    """
    try:
        from app.db.companies import fuzzy_search_companies, search_companies
        
        # Try fuzzy search first
        companies = fuzzy_search_companies(query, limit=limit)
        
        # If no results, try broader search
        if not companies:
            companies = search_companies(query, limit=limit)
        
        # Format response
        company_list = [
            {
                'id': company['id'],
                'name': company['company_name'],
                'nse_symbol': company.get('nse_symbol'),
                'bse_code': company.get('bse_code'),
                'industry': company.get('industry'),
                'market_cap': company.get('market_cap')
            }
            for company in companies
        ]
        
        return {
            'success': True,
            'companies': company_list,
            'total_companies': len(company_list),
            'query': query
        }
        
    except Exception as e:
        logger.error(f"Error searching companies: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint for news service."""
    return {
        'status': 'healthy',
        'service': 'news',
        'exa_configured': bool(news_service.exa_api_key)
    }
