"""
News Service
Handles fetching and processing news articles for companies using Exa API.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
from urllib.parse import urlparse
import re

from app.config import settings
from app.agent_tools.exa import exa_search, exa_live_search
from app.db.companies import (
    get_company_by_nse_symbol,
    get_company_by_bse_code,
    search_companies,
    fuzzy_search_companies,
    CompanyRow
)

logger = logging.getLogger(__name__)


class NewsService:
    """Service for fetching and processing company news articles."""
    
    def __init__(self):
        self.exa_api_key = settings.EXA_API_KEY
    
    def resolve_company(self, company_identifier: str) -> Optional[CompanyRow]:
        """
        Resolve company identifier to company data.
        Supports NSE symbol, BSE code, company name, or partial matches.
        """
        try:
            # Try exact NSE symbol match first
            company = get_company_by_nse_symbol(company_identifier.upper())
            if company:
                return company
            
            # Try BSE code match
            company = get_company_by_bse_code(company_identifier)
            if company:
                return company
            
            # Try fuzzy search for company name
            companies = fuzzy_search_companies(company_identifier, limit=1)
            if companies:
                return companies[0]
            
            # Try broader search
            companies = search_companies(company_identifier, limit=1)
            if companies:
                return companies[0]
            
            return None
            
        except Exception as e:
            logger.error(f"Error resolving company {company_identifier}: {str(e)}")
            return None
    
    def _clean_article_title(self, title: str) -> str:
        """Clean and format article title."""
        if not title:
            return "Untitled Article"
        
        # Remove extra whitespace
        title = re.sub(r'\s+', ' ', title.strip())
        
        # Limit length
        if len(title) > 120:
            title = title[:117] + "..."
        
        return title
    
    def _clean_article_content(self, content: str) -> str:
        """Clean and format article content for description."""
        if not content:
            return ""
        
        # Remove extra whitespace and newlines
        content = re.sub(r'\s+', ' ', content.strip())
        
        # Extract first meaningful sentence or paragraph
        sentences = content.split('. ')
        if sentences:
            # Take first 1-2 sentences for description
            description = '. '.join(sentences[:2])
            if not description.endswith('.'):
                description += '.'
        else:
            description = content
        
        # Limit length for description
        if len(description) > 200:
            description = description[:197] + "..."
        
        return description
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain name from URL for publisher."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            # Remove www. prefix
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except:
            return "Unknown"
    
    def _parse_exa_results(self, exa_results: str, company_name: str) -> List[Dict[str, Any]]:
        """
        Parse Exa search results and extract structured article data.
        """
        articles = []
        
        try:
            # Split results by numbered items (1., 2., etc.)
            lines = exa_results.split('\n')
            current_article = {}
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if line starts with a number (new article)
                if re.match(r'^\d+\.', line):
                    # Save previous article if exists
                    if current_article.get('title'):
                        articles.append(current_article)
                    
                    # Start new article
                    current_article = {}
                    
                    # Extract title and URL from the line
                    # Format: "1. Title (URL)"
                    match = re.match(r'^\d+\.\s*(.*?)\s*\((https?://[^)]+)\)', line)
                    if match:
                        title = match.group(1).strip()
                        url = match.group(2).strip()
                        
                        current_article = {
                            'title': self._clean_article_title(title),
                            'url': url,
                            'publisher': self._extract_domain(url),
                            'content': ''
                        }
                    else:
                        # Fallback: extract title without URL
                        title = re.sub(r'^\d+\.\s*', '', line)
                        current_article = {
                            'title': self._clean_article_title(title),
                            'url': '',
                            'publisher': 'Unknown',
                            'content': ''
                        }
                
                elif current_article and not line.startswith(('http://', 'https://')):
                    # This is content for the current article
                    if current_article['content']:
                        current_article['content'] += ' ' + line
                    else:
                        current_article['content'] = line
            
            # Don't forget the last article
            if current_article.get('title'):
                articles.append(current_article)
            
            # Clean up content for all articles
            for article in articles:
                article['description'] = self._clean_article_content(article.get('content', ''))
                # Remove full content as we only need description
                article.pop('content', None)
                
                # Add estimated date (since Exa doesn't always provide exact dates)
                # For demo, we'll use recent dates
                days_ago = len(articles) - articles.index(article)
                article_date = datetime.now() - timedelta(days=days_ago)
                article['published_at'] = article_date.strftime("%Y-%m-%d")
                article['date_display'] = self._format_date_display(article_date)
            
            return articles[:10]  # Limit to 10 most relevant articles
            
        except Exception as e:
            logger.error(f"Error parsing Exa results: {str(e)}")
            return []
    
    def _format_date_display(self, date: datetime) -> str:
        """Format date for display (e.g., 'May 25', 'Jan 28')."""
        return date.strftime("%b %d")
    
    def fetch_company_news(
        self, 
        company_identifier: str, 
        days_back: int = 30,
        max_articles: int = 10
    ) -> Dict[str, Any]:
        """
        Fetch news articles for a company.
        
        Args:
            company_identifier: Company name, NSE symbol, BSE code, or search term
            days_back: Number of days to look back for news
            max_articles: Maximum number of articles to return
            
        Returns:
            Dictionary with company info and news articles
        """
        try:
            # Resolve company
            company = self.resolve_company(company_identifier)
            if not company:
                return {
                    'success': False,
                    'error': f'Company not found: {company_identifier}',
                    'company': None,
                    'articles': []
                }
            
            company_name = company['company_name']
            logger.info(f"Fetching news for company: {company_name}")
            
            # Create search queries for better coverage
            search_queries = [
                f"{company_name} stock news earnings financial results",
                f"{company['nse_symbol']} share price news" if company.get('nse_symbol') else None,
                f"{company_name} quarterly results business news"
            ]
            
            # Remove None queries
            search_queries = [q for q in search_queries if q]
            
            all_articles = []
            
            # Search with multiple queries for comprehensive coverage
            for query in search_queries[:2]:  # Limit to 2 queries to avoid rate limits
                try:
                    # Use exa_live_search for more recent results
                    exa_results = exa_live_search.func(query, k=6, max_chars=1500)
                    parsed_articles = self._parse_exa_results(exa_results, company_name)
                    all_articles.extend(parsed_articles)
                except Exception as e:
                    logger.warning(f"Error with query '{query}': {str(e)}")
                    continue
            
            # Remove duplicates based on title similarity
            unique_articles = []
            seen_titles = set()
            
            for article in all_articles:
                title_key = article['title'].lower()[:50]  # First 50 chars for similarity
                if title_key not in seen_titles:
                    seen_titles.add(title_key)
                    unique_articles.append(article)
            
            # Sort by date (most recent first) and limit
            unique_articles = sorted(
                unique_articles, 
                key=lambda x: x.get('published_at', ''), 
                reverse=True
            )[:max_articles]
            
            return {
                'success': True,
                'company': {
                    'id': company['id'],
                    'name': company['company_name'],
                    'nse_symbol': company.get('nse_symbol'),
                    'bse_code': company.get('bse_code'),
                    'industry': company.get('industry'),
                    'market_cap': company.get('market_cap')
                },
                'articles': unique_articles,
                'total_articles': len(unique_articles),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error fetching news for {company_identifier}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'company': None,
                'articles': []
            }
    
    def get_trending_news(self, limit: int = 15) -> Dict[str, Any]:
        """
        Get trending financial news (not company-specific).
        """
        try:
            # Search for general market and financial news
            queries = [
                "Indian stock market news today financial",
                "NSE BSE market updates business news",
                "Indian economy financial markets news"
            ]
            
            all_articles = []
            
            for query in queries:
                try:
                    exa_results = exa_live_search.func(query, k=5, max_chars=1200)
                    parsed_articles = self._parse_exa_results(exa_results, "Market")
                    all_articles.extend(parsed_articles)
                except Exception as e:
                    logger.warning(f"Error with trending query '{query}': {str(e)}")
                    continue
            
            # Remove duplicates and sort
            unique_articles = []
            seen_titles = set()
            
            for article in all_articles:
                title_key = article['title'].lower()[:50]
                if title_key not in seen_titles:
                    seen_titles.add(title_key)
                    unique_articles.append(article)
            
            unique_articles = sorted(
                unique_articles,
                key=lambda x: x.get('published_at', ''),
                reverse=True
            )[:limit]
            
            return {
                'success': True,
                'articles': unique_articles,
                'total_articles': len(unique_articles),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error fetching trending news: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'articles': []
            }


# Singleton instance
news_service = NewsService()
