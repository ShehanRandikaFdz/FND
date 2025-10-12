"""
News Verification Module
Handles automatic verification of entered text against online news sources using NewsAPI
"""

import re
import time
from typing import Dict, List, Optional, Tuple
from collections import Counter
import streamlit as st

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    st.warning("sentence-transformers not available. Using basic keyword matching only.")

try:
    from news_apis.newsapi_client import NewsAPIClient
    NEWSAPI_AVAILABLE = True
except ImportError:
    NEWSAPI_AVAILABLE = False
    st.warning("NewsAPI client not available. Verification will be disabled.")


class NewsVerifier:
    """
    Verifies entered text against online news sources using NewsAPI
    Uses two-stage matching: keyword filtering (fast) + semantic similarity (accurate)
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize NewsVerifier with API client and similarity models"""
        self.api_client = None
        self.similarity_model = None
        self.tfidf_vectorizer = None
        
        # Initialize API client if available
        if NEWSAPI_AVAILABLE:
            try:
                self.api_client = NewsAPIClient(api_key=api_key)
                st.success("✅ NewsAPI client initialized")
            except Exception as e:
                st.warning(f"⚠️ NewsAPI client initialization failed: {e}")
        
        # Initialize similarity models if available
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # Use lightweight model for better performance
                self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
                st.success("✅ Semantic similarity models loaded")
            except Exception as e:
                st.warning(f"⚠️ Semantic similarity models failed to load: {e}")
    
    def _extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract important keywords from text for NewsAPI search"""
        # Clean text
        clean_text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = clean_text.split()
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 
            'above', 'below', 'between', 'among', 'is', 'are', 'was', 'were', 'be', 'been', 
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
        }
        
        # Filter out stop words and short words
        filtered_words = [w for w in words if len(w) > 2 and w not in stop_words]
        
        # Count word frequency and return most common
        word_counts = Counter(filtered_words)
        return [word for word, count in word_counts.most_common(max_keywords)]
    
    def _search_news(self, keywords: List[str], max_articles: int = 50) -> List[Dict]:
        """Search NewsAPI using extracted keywords"""
        if not self.api_client:
            return []
        
        try:
            # Combine keywords for search query
            query = ' '.join(keywords[:5])  # Use top 5 keywords to avoid query too long
            
            # Search for articles
            articles = self.api_client.search_everything(
                query=query,
                page_size=max_articles,
                sort_by='relevancy'
            )
            
            st.info(f"🔍 Searched for: '{query}' - Found {len(articles)} articles")
            return articles
            
        except Exception as e:
            st.error(f"❌ NewsAPI search failed: {e}")
            return []
    
    def _keyword_match(self, entered_text: str, articles: List[Dict], threshold: float = 0.3) -> List[Dict]:
        """Fast keyword-based filtering to narrow down articles"""
        if not articles:
            return []
        
        # Extract keywords from entered text
        entered_keywords = set(self._extract_keywords(entered_text, max_keywords=15))
        
        scored_articles = []
        
        for article in articles:
            # Combine title and description for matching
            article_text = f"{article.get('title', '')} {article.get('description', '')}"
            article_keywords = set(self._extract_keywords(article_text, max_keywords=15))
            
            # Calculate keyword overlap
            if entered_keywords and article_keywords:
                overlap = len(entered_keywords.intersection(article_keywords))
                keyword_score = overlap / len(entered_keywords.union(article_keywords))
                
                if keyword_score >= threshold:
                    article['keyword_score'] = keyword_score
                    scored_articles.append(article)
        
        # Sort by keyword score and return top matches
        scored_articles.sort(key=lambda x: x['keyword_score'], reverse=True)
        return scored_articles[:20]  # Return top 20 for semantic analysis
    
    def _semantic_similarity(self, entered_text: str, articles: List[Dict]) -> List[Dict]:
        """Calculate semantic similarity between entered text and articles"""
        if not articles or not self.similarity_model:
            return []
        
        try:
            # Prepare texts for comparison
            entered_clean = self._preprocess_for_similarity(entered_text)
            article_texts = []
            
            for article in articles:
                article_text = f"{article.get('title', '')} {article.get('description', '')}"
                article_texts.append(self._preprocess_for_similarity(article_text))
            
            # Calculate embeddings
            entered_embedding = self.similarity_model.encode([entered_clean])
            article_embeddings = self.similarity_model.encode(article_texts)
            
            # Calculate cosine similarity
            similarities = cosine_similarity(entered_embedding, article_embeddings)[0]
            
            # Add similarity scores to articles
            for i, article in enumerate(articles):
                article['similarity_score'] = float(similarities[i])
            
            # Filter by similarity threshold and sort
            filtered_articles = [a for a in articles if a['similarity_score'] > 0.4]
            filtered_articles.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            return filtered_articles[:5]  # Return top 5 matches
            
        except Exception as e:
            st.warning(f"⚠️ Semantic similarity calculation failed: {e}")
            return []
    
    def _preprocess_for_similarity(self, text: str) -> str:
        """Preprocess text for semantic similarity calculation"""
        if not text:
            return ""
        
        # Basic cleaning
        text = re.sub(r'http\S+|www\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text[:500]  # Limit length for better performance
    
    def verify_news(self, text: str) -> Dict:
        """
        Main verification pipeline:
        1. Extract keywords from entered text
        2. Search NewsAPI
        3. Stage 1: Keyword filtering (fast)
        4. Stage 2: Semantic similarity on top matches (accurate)
        """
        if not self.api_client:
            return {
                'found_online': False,
                'error': 'NewsAPI client not available',
                'best_match': None,
                'similarity_score': 0,
                'all_matches': []
            }
        
        if not text or len(text.strip()) < 10:
            return {
                'found_online': False,
                'error': 'Text too short for verification',
                'best_match': None,
                'similarity_score': 0,
                'all_matches': []
            }
        
        try:
            with st.spinner("🔍 Verifying against online sources..."):
                # Step 1: Extract keywords
                keywords = self._extract_keywords(text)
                st.info(f"📝 Extracted keywords: {', '.join(keywords[:5])}")
                
                # Step 2: Search NewsAPI
                articles = self._search_news(keywords)
                
                if not articles:
                    return {
                        'found_online': False,
                        'error': 'No articles found in search',
                        'best_match': None,
                        'similarity_score': 0,
                        'all_matches': []
                    }
                
                # Step 3: Keyword filtering
                keyword_matches = self._keyword_match(text, articles, threshold=0.2)
                st.info(f"🎯 Keyword matches: {len(keyword_matches)} articles")
                
                # Step 4: Semantic similarity
                if keyword_matches and self.similarity_model:
                    semantic_matches = self._semantic_similarity(text, keyword_matches)
                    st.info(f"🧠 Semantic matches: {len(semantic_matches)} articles")
                else:
                    # Fallback to keyword matches if semantic model not available
                    semantic_matches = keyword_matches[:3]
                    for article in semantic_matches:
                        article['similarity_score'] = article.get('keyword_score', 0)
                
                # Return results
                if semantic_matches:
                    best_match = semantic_matches[0]
                    return {
                        'found_online': True,
                        'best_match': best_match,
                        'similarity_score': best_match['similarity_score'],
                        'all_matches': semantic_matches,
                        'search_keywords': keywords[:5],
                        'total_articles_found': len(articles)
                    }
                else:
                    return {
                        'found_online': False,
                        'error': 'No similar articles found',
                        'best_match': None,
                        'similarity_score': 0,
                        'all_matches': [],
                        'search_keywords': keywords[:5],
                        'total_articles_found': len(articles)
                    }
        
        except Exception as e:
            st.error(f"❌ Verification failed: {e}")
            return {
                'found_online': False,
                'error': f'Verification error: {str(e)}',
                'best_match': None,
                'similarity_score': 0,
                'all_matches': []
            }
    
    def get_verification_summary(self, verification_result: Dict) -> str:
        """Generate a human-readable summary of verification results"""
        if not verification_result.get('found_online', False):
            return "❌ **Not found online** - No matching articles detected"
        
        similarity = verification_result.get('similarity_score', 0)
        best_match = verification_result.get('best_match', {})
        
        if similarity > 0.8:
            return f"✅ **Highly verified** - Found matching article from {best_match.get('source', {}).get('name', 'Unknown')} (Similarity: {similarity:.1%})"
        elif similarity > 0.6:
            return f"✅ **Likely verified** - Found similar article from {best_match.get('source', {}).get('name', 'Unknown')} (Similarity: {similarity:.1%})"
        elif similarity > 0.4:
            return f"⚠️ **Partially verified** - Found related article from {best_match.get('source', {}).get('name', 'Unknown')} (Similarity: {similarity:.1%})"
        else:
            return f"❓ **Weak match** - Found loosely related article (Similarity: {similarity:.1%})"
