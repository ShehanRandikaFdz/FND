"""
News Verifier for Flask Application
Verifies news against online sources using NewsAPI and semantic similarity
"""

import os
import re
from typing import Dict, List, Optional, Tuple
import requests
from config import Config

try:
    from sentence_transformers import SentenceTransformer
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    print("sentence-transformers not available - using keyword matching only")

class NewsVerifier:
    """Verify news articles against online sources"""
    
    def __init__(self):
        self.newsapi_key = Config.NEWSAPI_KEY
        self.similarity_model = None
        
        # Initialize semantic similarity model if available
        if SEMANTIC_AVAILABLE:
            try:
                self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("✅ Semantic similarity model loaded")
            except Exception as e:
                print(f"⚠️ Failed to load semantic model: {e}")
                self.similarity_model = None
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract key terms from text for searching"""
        if not text:
            return []
        
        # Clean text
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text.split()
        
        # Filter out common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these',
            'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }
        
        # Extract meaningful words (longer than 3 characters, not stop words)
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]
        
        # Return most frequent keywords
        from collections import Counter
        keyword_counts = Counter(keywords)
        return [word for word, count in keyword_counts.most_common(max_keywords)]
    
    def search_newsapi(self, query: str, page_size: int = 20) -> List[Dict]:
        """Search NewsAPI for articles matching the query"""
        if not self.newsapi_key:
            print("⚠️ NewsAPI key not configured")
            return []
        
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': query,
                'apiKey': self.newsapi_key,
                'pageSize': page_size,
                'sortBy': 'relevancy',
                'language': 'en'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            articles = data.get('articles', [])
            
            return articles
            
        except Exception as e:
            print(f"❌ NewsAPI search failed: {e}")
            return []
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        if not self.similarity_model:
            # Fallback to simple keyword overlap
            return self._keyword_similarity(text1, text2)
        
        try:
            embeddings = self.similarity_model.encode([text1, text2])
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except Exception as e:
            print(f"⚠️ Semantic similarity failed: {e}")
            return self._keyword_similarity(text1, text2)
    
    def _keyword_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity based on keyword overlap"""
        keywords1 = set(self.extract_keywords(text1))
        keywords2 = set(self.extract_keywords(text2))
        
        if not keywords1 or not keywords2:
            return 0.0
        
        intersection = keywords1.intersection(keywords2)
        union = keywords1.union(keywords2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def find_matching_articles(self, input_text: str, articles: List[Dict]) -> List[Dict]:
        """Find articles that match the input text"""
        matches = []
        
        for article in articles:
            title = article.get('title', '')
            description = article.get('description', '')
            content = article.get('content', '')
            
            # Combine article text
            article_text = f"{title} {description} {content}".strip()
            
            if not article_text:
                continue
            
            # Calculate similarity
            similarity = self.calculate_similarity(input_text, article_text)
            
            if similarity > 0.3:  # Minimum similarity threshold
                article['similarity_score'] = similarity
                matches.append(article)
        
        # Sort by similarity score
        matches.sort(key=lambda x: x['similarity_score'], reverse=True)
        return matches
    
    def verify_news(self, text: str) -> Dict:
        """Verify news text against online sources"""
        try:
            # Extract keywords for searching
            keywords = self.extract_keywords(text)
            
            if not keywords:
                return {
                    'found_online': False,
                    'error': 'No keywords extracted from text',
                    'best_match': None,
                    'similarity_score': 0,
                    'all_matches': []
                }
            
            # Create search query from top keywords
            query = ' '.join(keywords[:5])
            
            # Search NewsAPI
            articles = self.search_newsapi(query)
            
            if not articles:
                return {
                    'found_online': False,
                    'error': 'No articles found in NewsAPI',
                    'best_match': None,
                    'similarity_score': 0,
                    'all_matches': []
                }
            
            # Find matching articles
            matches = self.find_matching_articles(text, articles)
            
            if matches:
                best_match = matches[0]
                return {
                    'found_online': True,
                    'best_match': best_match,
                    'similarity_score': best_match['similarity_score'],
                    'all_matches': matches[:5],  # Top 5 matches
                    'total_matches': len(matches),
                    'search_query': query
                }
            else:
                return {
                    'found_online': False,
                    'error': 'No similar articles found',
                    'best_match': None,
                    'similarity_score': 0,
                    'all_matches': [],
                    'search_query': query
                }
                
        except Exception as e:
            return {
                'found_online': False,
                'error': str(e),
                'best_match': None,
                'similarity_score': 0,
                'all_matches': []
            }