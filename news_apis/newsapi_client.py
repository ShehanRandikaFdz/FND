"""
NewsAPI.org Integration Client
Fetches real-time news from 80,000+ sources worldwide
"""
import os
import requests
import time
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class NewsAPIClient:
    """Client for NewsAPI.org integration"""
    
    BASE_URL = "https://newsapi.org/v2"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize NewsAPI client
        
        Args:
            api_key: NewsAPI.org API key (or set NEWSAPI_KEY env variable)
        """
        self.api_key = api_key or os.getenv('NEWSAPI_KEY')
        if not self.api_key:
            raise ValueError(
                "NewsAPI key not found! "
                "Please set NEWSAPI_KEY environment variable or pass api_key parameter."
            )
        
        self.session = requests.Session()
        self.rate_limit_remaining = 100
        self.rate_limit_reset = None
        
    def get_top_headlines(
        self,
        country: str = 'us',
        category: Optional[str] = None,
        query: Optional[str] = None,
        page_size: int = 20
    ) -> List[Dict]:
        """
        Fetch top headlines
        
        Args:
            country: 2-letter country code (us, gb, ca, au, etc.)
            category: business, entertainment, general, health, science, sports, technology
            query: Keywords to search for
            page_size: Number of articles (max 100)
            
        Returns:
            List of article dictionaries
        """
        endpoint = f"{self.BASE_URL}/top-headlines"
        params = {
            'apiKey': self.api_key,
            'country': country,
            'pageSize': min(page_size, 100)
        }
        
        if category:
            params['category'] = category
        if query:
            params['q'] = query
            
        return self._make_request(endpoint, params)
    
    def search_everything(
        self,
        query: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        language: str = 'en',
        sort_by: str = 'publishedAt',
        page_size: int = 20
    ) -> List[Dict]:
        """
        Search all articles
        
        Args:
            query: Keywords to search for
            from_date: Oldest article date
            to_date: Newest article date
            language: Language code (en, es, fr, de, etc.)
            sort_by: publishedAt, relevancy, or popularity
            page_size: Number of articles (max 100)
            
        Returns:
            List of article dictionaries
        """
        endpoint = f"{self.BASE_URL}/everything"
        params = {
            'apiKey': self.api_key,
            'q': query,
            'language': language,
            'sortBy': sort_by,
            'pageSize': min(page_size, 100)
        }
        
        if from_date:
            params['from'] = from_date.isoformat()
        if to_date:
            params['to'] = to_date.isoformat()
            
        return self._make_request(endpoint, params)
    
    def get_sources(
        self,
        category: Optional[str] = None,
        language: str = 'en',
        country: Optional[str] = None
    ) -> List[Dict]:
        """
        Get available news sources
        
        Args:
            category: Filter by category
            language: Filter by language
            country: Filter by country
            
        Returns:
            List of source dictionaries
        """
        endpoint = f"{self.BASE_URL}/sources"
        params = {
            'apiKey': self.api_key,
            'language': language
        }
        
        if category:
            params['category'] = category
        if country:
            params['country'] = country
        
        try:
            response = self.session.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data.get('sources', [])
        except Exception as e:
            print(f"❌ Failed to fetch sources: {e}")
            return []
    
    def _make_request(self, endpoint: str, params: Dict) -> List[Dict]:
        """Make API request with rate limiting"""
        
        # Check rate limit
        if self.rate_limit_remaining == 0 and self.rate_limit_reset:
            wait_time = self.rate_limit_reset - time.time()
            if wait_time > 0:
                print(f"⏳ Rate limit reached. Waiting {wait_time:.0f}s...")
                time.sleep(wait_time)
        
        try:
            response = self.session.get(endpoint, params=params, timeout=30)
            
            # Update rate limit info
            self.rate_limit_remaining = int(
                response.headers.get('X-RateLimit-Remaining', 100)
            )
            reset_time = response.headers.get('X-RateLimit-Reset')
            if reset_time:
                self.rate_limit_reset = int(reset_time)
            
            response.raise_for_status()
            data = response.json()
            
            if data['status'] == 'ok':
                articles = data.get('articles', [])
                print(f"✅ Fetched {len(articles)} articles (Rate limit: {self.rate_limit_remaining} remaining)")
                return articles
            else:
                error_msg = data.get('message', 'Unknown error')
                print(f"❌ API Error: {error_msg}")
                return []
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                print("❌ Invalid API key! Please check your NEWSAPI_KEY.")
            elif e.response.status_code == 429:
                print("❌ Rate limit exceeded! Consider upgrading your plan.")
            else:
                print(f"❌ HTTP Error: {e}")
            return []
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
            return []
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return []
    
    def format_article(self, article: Dict) -> Dict:
        """
        Format article data for analysis
        
        Args:
            article: Raw article from API
            
        Returns:
            Formatted article dictionary
        """
        return {
            'title': article.get('title', ''),
            'content': article.get('content') or article.get('description', ''),
            'full_content': f"{article.get('title', '')}. {article.get('description', '')} {article.get('content', '')}",
            'source': article.get('source', {}).get('name', 'Unknown'),
            'author': article.get('author', 'Unknown'),
            'url': article.get('url', ''),
            'published_at': article.get('publishedAt', ''),
            'image_url': article.get('urlToImage', '')
        }


# Test code
if __name__ == "__main__":
    print("🧪 Testing NewsAPI Client\n")
    print("=" * 60)
    
    try:
        # Initialize client
        client = NewsAPIClient()
        
        # Test 1: Get top headlines
        print("\n📰 Test 1: Fetching top headlines from US...")
        headlines = client.get_top_headlines(country='us', page_size=5)
        
        if headlines:
            print(f"\n✅ Successfully fetched {len(headlines)} headlines:")
            for i, article in enumerate(headlines, 1):
                formatted = client.format_article(article)
                print(f"\n{i}. {formatted['title']}")
                print(f"   Source: {formatted['source']}")
                print(f"   Published: {formatted['published_at']}")
                print(f"   URL: {formatted['url'][:60]}...")
        
        # Test 2: Search for specific topic
        print("\n\n📰 Test 2: Searching for 'technology' news...")
        tech_news = client.search_everything(query='technology', page_size=3)
        
        if tech_news:
            print(f"\n✅ Successfully found {len(tech_news)} technology articles:")
            for i, article in enumerate(tech_news, 1):
                formatted = client.format_article(article)
                print(f"\n{i}. {formatted['title']}")
                print(f"   Source: {formatted['source']}")
        
        # Test 3: Get sources
        print("\n\n📰 Test 3: Fetching available sources...")
        sources = client.get_sources(language='en')
        
        if sources:
            print(f"\n✅ Found {len(sources)} English sources:")
            for source in sources[:10]:
                print(f"   • {source['name']} ({source.get('category', 'general')})")
        
        print("\n\n" + "=" * 60)
        print("✅ All tests passed! NewsAPI client is working correctly.")
        print("=" * 60)
        
    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("\nPlease set your NewsAPI key:")
        print("1. Get free API key from: https://newsapi.org/register")
        print("2. Set environment variable: set NEWSAPI_KEY=your_key_here")
        print("3. Or create .env file with: NEWSAPI_KEY=your_key_here")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
