"""
NewsAPI Client
Simplified client for NewsAPI integration
"""

import requests
from typing import List, Dict, Optional
import time

class NewsAPIClient:
    """Simple NewsAPI client for fetching news articles"""
    
    def __init__(self, api_key: str):
        """Initialize NewsAPI client"""
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"
        self.session = requests.Session()
        # NewsAPI accepts both 'X-Api-Key' and 'Authorization: Bearer', prefer Authorization
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'User-Agent': 'Fake-News-Detector/1.0'
        })
    
    def get_top_headlines(
        self, 
        country: Optional[str] = None,
        category: Optional[str] = None,
        page_size: int = 10,
        page: int = 1
    ) -> List[Dict]:
        """Get top headlines"""
        try:
            params = {
                'pageSize': min(page_size, 100),  # API limit
                'page': page
            }
            
            if country:
                params['country'] = country
            if category:
                params['category'] = category
            
            response = self.session.get(f"{self.base_url}/top-headlines", params=params)
            response.raise_for_status()
            
            data = response.json()
            return data.get('articles', [])
            
        except Exception as e:
            print(f"❌ Error fetching top headlines: {e}")
            return []
    
    def search_everything(
        self,
        query: str,
        page_size: int = 10,
        page: int = 1,
        sort_by: str = 'relevancy'
    ) -> List[Dict]:
        """Search for articles"""
        try:
            params = {
                'q': query,
                'pageSize': min(page_size, 100),  # API limit
                'page': page,
                'sortBy': sort_by,
                'language': 'en'
            }
            
            response = self.session.get(f"{self.base_url}/everything", params=params)
            response.raise_for_status()
            
            data = response.json()
            return data.get('articles', [])
            
        except Exception as e:
            print(f"❌ Error searching articles: {e}")
            return []
    
    def get_everything(
        self,
        query: str,
        language: str = 'en',
        page_size: int = 10,
        page: int = 1
    ) -> List[Dict]:
        """Alias for search_everything"""
        return self.search_everything(query, page_size, page)