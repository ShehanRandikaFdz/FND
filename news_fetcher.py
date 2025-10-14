"""
News Fetcher for Fake News Detector
Fetches news from NewsAPI and analyzes credibility
"""

import time
from datetime import datetime
from typing import List, Dict, Optional
import hashlib

from news_apis.newsapi_client import NewsAPIClient
from utils.model_loader import ModelLoader
from utils.predictor import UnifiedPredictor
from config import Config


class NewsFetcher:
    """Fetches and analyzes news articles"""
    
    def __init__(self):
        """Initialize news fetcher with API client and analyzer"""
        print("Initializing News Fetcher...")
        
        # Validate configuration
        if not Config.NEWSAPI_KEY:
            raise ValueError("NewsAPI key not configured! Please check your .env file.")
        
        # Initialize API client
        print("Connecting to NewsAPI...")
        self.api_client = NewsAPIClient(api_key=Config.NEWSAPI_KEY)
        
        # Initialize ML components (lazy loading)
        self.model_loader = None
        self.predictor = None
        
        # Track processed articles to avoid duplicates
        self.seen_urls = set()
        self.seen_hashes = set()
        
        print("News Fetcher initialized successfully!")
    
    def _load_ml_models(self):
        """Load ML models lazily"""
        if self.model_loader is None:
            print("Loading ML models...")
            self.model_loader = ModelLoader()
            self.model_loader.load_svm_model()
            self.model_loader.load_lstm_model()
            self.model_loader.load_bert_model()
            
            self.predictor = UnifiedPredictor(self.model_loader)
            print("ML models loaded!")
    
    def fetch_and_analyze(
        self,
        country: Optional[str] = None,
        category: Optional[str] = None,
        query: Optional[str] = None,
        page_size: Optional[int] = None,
        page: int = 1
    ) -> List[Dict]:
        """
        Fetch news and analyze credibility with pagination support
        
        Args:
            country: Country code (e.g., 'us', 'gb')
            category: News category (e.g., 'general', 'technology')
            query: Search query
            page_size: Number of articles to fetch per page
            page: Page number for pagination (default: 1)
            
        Returns:
            List of analyzed articles with credibility scores
        """
        try:
            # Load ML models
            self._load_ml_models()
            
            # Fetch articles from NewsAPI with pagination
            print(f"Fetching articles (page {page})...")
            if query:
                articles = self.api_client.search_articles(
                    query=query,
                    page_size=page_size or 10,
                    page=page
                )
            else:
                articles = self.api_client.get_top_headlines(
                    country=country,
                    category=category,
                    page_size=page_size or 10,
                    page=page
                )
            
            if not articles:
                print("No articles found")
                return []
            
            # Analyze each article
            analyzed_articles = []
            for article in articles:
                try:
                    analyzed = self._analyze_article(article)
                    if analyzed:
                        analyzed_articles.append(analyzed)
                except Exception as e:
                    print(f"Error analyzing article: {e}")
                    continue
            
            print(f"Successfully analyzed {len(analyzed_articles)} articles")
            return analyzed_articles
            
        except Exception as e:
            print(f"Error fetching news: {e}")
            return []
    
    def _analyze_article(self, article: Dict) -> Optional[Dict]:
        """Analyze a single article for credibility"""
        try:
            # Check for duplicates
            article_hash = hashlib.md5(article.get('url', '').encode()).hexdigest()
            if article_hash in self.seen_hashes:
                return None
            
            self.seen_hashes.add(article_hash)
            
            # Prepare text for analysis
            title = article.get('title', '')
            description = article.get('description', '')
            content = article.get('content', '')
            
            # Combine text (prefer content, fallback to description)
            text_to_analyze = content or description or title
            
            if not text_to_analyze or len(text_to_analyze.strip()) < 10:
                return None
            
            # Make prediction
            result = self.predictor.ensemble_predict_majority(text_to_analyze)
            
            # Calculate credibility score
            prediction = result.get('final_prediction', 'UNKNOWN')
            confidence = result.get('confidence', 0)
            
            if prediction == 'TRUE':
                credibility_score = confidence / 100.0
            else:
                credibility_score = (100 - confidence) / 100.0
            
            # Get individual model results
            individual_results = result.get('individual_results', {})
            
            return {
                'title': title,
                'description': description,
                'content': content,
                'url': article.get('url', ''),
                'source': article.get('source', {}).get('name', 'Unknown'),
                'author': article.get('author', 'Unknown'),
                'published_at': article.get('publishedAt', ''),
                'image_url': article.get('urlToImage', ''),
                'credibility_score': credibility_score,
                'confidence': confidence / 100.0,
                'prediction': prediction,
                'individual_predictions': individual_results,
                'analyzed_at': datetime.now().isoformat(),
                'text_analyzed': text_to_analyze[:200] + "..." if len(text_to_analyze) > 200 else text_to_analyze
            }
            
        except Exception as e:
            print(f"Error analyzing article: {e}")
            return None
    
    def search_and_analyze(
        self,
        query: str,
        page_size: Optional[int] = None
    ) -> List[Dict]:
        """
        Search for articles by query and analyze them
        
        Args:
            query: Search query
            page_size: Number of articles to fetch
            
        Returns:
            List of analyzed articles
        """
        return self.fetch_and_analyze(query=query, page_size=page_size)
    
    def get_statistics(self, articles: List[Dict]) -> Dict:
        """Get statistics from analyzed articles"""
        if not articles:
            return {
                'total': 0,
                'fake': 0,
                'suspicious': 0,
                'credible': 0,
                'fake_percentage': 0,
                'suspicious_percentage': 0,
                'credible_percentage': 0,
                'average_credibility': 0
            }
        
        total = len(articles)
        fake = sum(1 for a in articles if a['credibility_score'] < 0.3)
        suspicious = sum(1 for a in articles if 0.3 <= a['credibility_score'] < 0.7)
        credible = sum(1 for a in articles if a['credibility_score'] >= 0.7)
        
        avg_credibility = sum(a['credibility_score'] for a in articles) / total
        
        return {
            'total': total,
            'fake': fake,
            'suspicious': suspicious,
            'credible': credible,
            'fake_percentage': (fake / total) * 100,
            'suspicious_percentage': (suspicious / total) * 100,
            'credible_percentage': (credible / total) * 100,
            'average_credibility': avg_credibility
        }
