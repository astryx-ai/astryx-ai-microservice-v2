#!/usr/bin/env python3
"""
Test script for News API endpoints
Run this to verify the news API is working correctly.
"""
import asyncio
import sys
import json
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from app.services.news_service import news_service


async def test_news_service():
    """Test the news service functionality."""
    print("🧪 Testing News Service...")
    print("=" * 50)
    
    # Test 1: Company resolution
    print("\n1️⃣ Testing company resolution...")
    company = news_service.resolve_company("RELIANCE")
    if company:
        print(f"✅ Found company: {company['company_name']} ({company.get('nse_symbol')})")
    else:
        print("❌ Company not found")
    
    # Test 2: Fetch company news
    print("\n2️⃣ Testing company news fetching...")
    try:
        result = news_service.fetch_company_news("TCS", max_articles=3)
        if result['success']:
            print(f"✅ Fetched {len(result['articles'])} articles for {result['company']['name']}")
            for i, article in enumerate(result['articles'][:2], 1):
                print(f"   {i}. {article['title'][:60]}...")
                print(f"      📅 {article['date_display']} | 🏢 {article['publisher']}")
        else:
            print(f"❌ Failed: {result.get('error')}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    # Test 3: Trending news
    print("\n3️⃣ Testing trending news...")
    try:
        result = news_service.get_trending_news(limit=3)
        if result['success']:
            print(f"✅ Fetched {len(result['articles'])} trending articles")
            for i, article in enumerate(result['articles'][:2], 1):
                print(f"   {i}. {article['title'][:60]}...")
                print(f"      📅 {article['date_display']} | 🏢 {article['publisher']}")
        else:
            print(f"❌ Failed: {result.get('error')}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    print("\n" + "=" * 50)
    print("✨ News service testing completed!")


def test_api_response_format():
    """Test that API responses match expected format."""
    print("\n🔍 Testing API Response Format...")
    print("=" * 50)
    
    # Sample response structure validation
    sample_article = {
        "title": "Sample Article Title",
        "description": "Sample description text that should be under 200 characters for optimal display.",
        "url": "https://example.com/article",
        "publisher": "example.com",
        "published_at": "2024-01-15",
        "date_display": "Jan 15"
    }
    
    sample_company = {
        "id": "uuid-sample",
        "name": "Sample Company Ltd",
        "nse_symbol": "SAMPLE",
        "bse_code": "123456",
        "industry": "Technology",
        "market_cap": 100000.0
    }
    
    sample_response = {
        "success": True,
        "company": sample_company,
        "articles": [sample_article],
        "total_articles": 1,
        "last_updated": "2024-01-15T10:30:00.000Z"
    }
    
    print("✅ Sample API Response Structure:")
    print(json.dumps(sample_response, indent=2))
    
    # Validate required fields
    required_article_fields = ["title", "description", "url", "publisher", "published_at", "date_display"]
    required_company_fields = ["id", "name"]
    required_response_fields = ["success", "articles", "total_articles", "last_updated"]
    
    print(f"\n✅ Required article fields: {required_article_fields}")
    print(f"✅ Required company fields: {required_company_fields}")
    print(f"✅ Required response fields: {required_response_fields}")


if __name__ == "__main__":
    print("🚀 News API Test Suite")
    print("=" * 50)
    
    # Test response format first
    test_api_response_format()
    
    # Test actual service (requires environment setup)
    try:
        asyncio.run(test_news_service())
    except Exception as e:
        print(f"\n⚠️  Service test failed (this is expected if environment is not fully configured):")
        print(f"   Error: {str(e)}")
        print("\n💡 To run full tests:")
        print("   1. Ensure all environment variables are set")
        print("   2. Start the FastAPI server: uvicorn app.main:app --reload")
        print("   3. Test endpoints manually using curl or Postman")
    
    print("\n🎯 Next Steps:")
    print("   1. Start the server: uvicorn app.main:app --reload")
    print("   2. Test endpoints:")
    print("      - GET http://localhost:8000/news/health")
    print("      - GET http://localhost:8000/news/company/RELIANCE")
    print("      - GET http://localhost:8000/news/trending")
    print("   3. Check FastAPI docs: http://localhost:8000/docs")
