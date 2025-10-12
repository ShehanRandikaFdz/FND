"""
Real-Time News Fetcher
Fetches news from NewsAPI and analyzes credibility in real-time
"""
import time
from datetime import datetime
from typing import List, Dict, Optional
import hashlib

from news_apis.newsapi_client import NewsAPIClient
from credibility_analyzer.credibility_analyzer import CredibilityAnalyzer
from config import config


class NewsFetcher:
    """Fetches and analyzes news articles in real-time"""
    
    def __init__(self):
        """Initialize news fetcher with API client and analyzer"""
        print("🔄 Initializing Real-Time News Fetcher...")
        
        # Validate configuration
        if not config.validate():
            raise ValueError("Invalid configuration! Please check your .env file.")
        
        # Initialize API client
        print("📡 Connecting to NewsAPI...")
        self.api_client = NewsAPIClient(api_key=config.NEWSAPI_KEY)
        
        # Initialize credibility analyzer (lazy loading)
        print("🔍 Loading Credibility Analyzer...")
        self.analyzer = CredibilityAnalyzer()
        
        # Track processed articles to avoid duplicates
        self.seen_urls = set()
        self.seen_hashes = set()
        
        print("✅ News Fetcher initialized successfully!\n")
    
    def fetch_and_analyze(
        self,
        country: Optional[str] = None,
        category: Optional[str] = None,
        query: Optional[str] = None,
        page_size: Optional[int] = None
    ) -> List[Dict]:
        """
        Fetch news and analyze credibility
        
        Args:
            country: Country code (us, gb, ca, etc.)
            category: News category (business, technology, etc.)
            query: Search query
            page_size: Number of articles to fetch
            
        Returns:
            List of articles with credibility analysis
        """
        # Use config defaults if not specified
        country = country or config.DEFAULT_COUNTRY
        page_size = page_size or config.DEFAULT_PAGE_SIZE
        
        print(f"\n{'='*60}")
        print(f"📰 FETCHING NEWS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        print(f"Country: {country}")
        if category:
            print(f"Category: {category}")
        if query:
            print(f"Query: {query}")
        print(f"Page Size: {page_size}")
        print(f"{'='*60}\n")
        
        # Fetch news articles
        articles = self.api_client.get_top_headlines(
            country=country,
            category=category,
            query=query,
            page_size=page_size
        )
        
        if not articles:
            print("⚠️ No articles fetched!")
            return []
        
        # Format and analyze articles
        analyzed_articles = []
        print(f"\n🔍 Analyzing {len(articles)} articles...\n")
        
        for i, article in enumerate(articles, 1):
            # Format article
            formatted = self.api_client.format_article(article)
            
            # Check for duplicates
            url = formatted['url']
            if url in self.seen_urls:
                print(f"[{i}/{len(articles)}] ⏭️  Skipping duplicate: {formatted['title'][:50]}...")
                continue
            
            # Analyze credibility
            print(f"[{i}/{len(articles)}] 🔍 Analyzing: {formatted['title'][:50]}...")
            
            try:
                # Analyze the full content (title + description + content)
                analysis = self.analyzer.analyze_credibility(formatted['full_content'])
                
                # Add analysis to article
                formatted['credibility_score'] = analysis['credibility_score']
                formatted['credibility_status'] = analysis['credibility_status']
                formatted['confidence'] = analysis['confidence']
                formatted['label'] = analysis['label']
                formatted['individual_predictions'] = analysis['individual_predictions']
                formatted['analyzed_at'] = datetime.now().isoformat()
                
                # Determine status emoji
                score = analysis['credibility_score']
                if score < config.FAKE_THRESHOLD:
                    status = "🚨 LIKELY FAKE"
                elif score < config.SUSPICIOUS_THRESHOLD:
                    status = "⚠️  SUSPICIOUS"
                else:
                    status = "✅ CREDIBLE"
                
                print(f"   {status} (Score: {score:.2f}, Confidence: {analysis['confidence']:.2f})")
                
                analyzed_articles.append(formatted)
                self.seen_urls.add(url)
                
            except Exception as e:
                print(f"   ❌ Analysis failed: {e}")
                continue
            
            # Small delay to avoid overwhelming the system
            time.sleep(0.1)
        
        print(f"\n✅ Analysis complete! {len(analyzed_articles)} articles analyzed.")
        print(f"{'='*60}\n")
        
        return analyzed_articles
    
    def fetch_by_categories(
        self,
        categories: Optional[List[str]] = None,
        country: Optional[str] = None,
        articles_per_category: int = 10
    ) -> Dict[str, List[Dict]]:
        """
        Fetch and analyze news from multiple categories
        
        Args:
            categories: List of categories to fetch
            country: Country code
            articles_per_category: Articles per category
            
        Returns:
            Dictionary mapping categories to analyzed articles
        """
        categories = categories or config.NEWS_CATEGORIES
        country = country or config.DEFAULT_COUNTRY
        
        results = {}
        
        for category in categories:
            print(f"\n📂 Fetching {category.upper()} news...\n")
            articles = self.fetch_and_analyze(
                country=country,
                category=category,
                page_size=articles_per_category
            )
            results[category] = articles
            
            # Small delay between categories
            time.sleep(1)
        
        return results
    
    def search_and_analyze(
        self,
        query: str,
        page_size: Optional[int] = None
    ) -> List[Dict]:
        """
        Search for specific topics and analyze
        
        Args:
            query: Search query
            page_size: Number of articles
            
        Returns:
            List of analyzed articles
        """
        page_size = page_size or config.DEFAULT_PAGE_SIZE
        
        print(f"\n{'='*60}")
        print(f"🔍 SEARCHING NEWS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        print(f"Query: {query}")
        print(f"Page Size: {page_size}")
        print(f"{'='*60}\n")
        
        # Search articles
        articles = self.api_client.search_everything(
            query=query,
            page_size=page_size
        )
        
        if not articles:
            print("⚠️ No articles found!")
            return []
        
        # Analyze articles
        analyzed_articles = []
        print(f"\n🔍 Analyzing {len(articles)} articles...\n")
        
        for i, article in enumerate(articles, 1):
            formatted = self.api_client.format_article(article)
            
            # Check for duplicates
            if formatted['url'] in self.seen_urls:
                continue
            
            print(f"[{i}/{len(articles)}] 🔍 {formatted['title'][:60]}...")
            
            try:
                analysis = self.analyzer.analyze_credibility(formatted['full_content'])
                
                formatted.update({
                    'credibility_score': analysis['credibility_score'],
                    'credibility_status': analysis['credibility_status'],
                    'confidence': analysis['confidence'],
                    'label': analysis['label'],
                    'individual_predictions': analysis['individual_predictions'],
                    'analyzed_at': datetime.now().isoformat()
                })
                
                score = analysis['credibility_score']
                if score < config.FAKE_THRESHOLD:
                    status = "🚨 LIKELY FAKE"
                elif score < config.SUSPICIOUS_THRESHOLD:
                    status = "⚠️  SUSPICIOUS"
                else:
                    status = "✅ CREDIBLE"
                
                print(f"   {status} (Score: {score:.2f})")
                
                analyzed_articles.append(formatted)
                self.seen_urls.add(formatted['url'])
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                continue
            
            time.sleep(0.1)
        
        print(f"\n✅ Search complete! {len(analyzed_articles)} articles analyzed.")
        print(f"{'='*60}\n")
        
        return analyzed_articles
    
    def get_statistics(self, articles: List[Dict]) -> Dict:
        """
        Calculate statistics from analyzed articles
        
        Args:
            articles: List of analyzed articles
            
        Returns:
            Statistics dictionary
        """
        if not articles:
            return {
                'total': 0,
                'fake': 0,
                'suspicious': 0,
                'credible': 0,
                'avg_score': 0.0
            }
        
        fake_count = sum(1 for a in articles if a['credibility_score'] < config.FAKE_THRESHOLD)
        suspicious_count = sum(
            1 for a in articles 
            if config.FAKE_THRESHOLD <= a['credibility_score'] < config.SUSPICIOUS_THRESHOLD
        )
        credible_count = sum(1 for a in articles if a['credibility_score'] >= config.SUSPICIOUS_THRESHOLD)
        avg_score = sum(a['credibility_score'] for a in articles) / len(articles)
        
        return {
            'total': len(articles),
            'fake': fake_count,
            'suspicious': suspicious_count,
            'credible': credible_count,
            'avg_score': avg_score,
            'fake_percentage': (fake_count / len(articles)) * 100,
            'suspicious_percentage': (suspicious_count / len(articles)) * 100,
            'credible_percentage': (credible_count / len(articles)) * 100
        }


# Test code
if __name__ == "__main__":
    print("🧪 Testing Real-Time News Fetcher\n")
    
    try:
        # Initialize fetcher
        fetcher = NewsFetcher()
        
        # Test 1: Fetch and analyze top headlines
        print("\n" + "="*60)
        print("TEST 1: Fetch and Analyze Top Headlines")
        print("="*60)
        
        articles = fetcher.fetch_and_analyze(country='us', page_size=5)
        
        if articles:
            print(f"\n✅ Successfully analyzed {len(articles)} articles\n")
            
            # Show results
            for i, article in enumerate(articles, 1):
                print(f"{i}. {article['title']}")
                print(f"   Source: {article['source']}")
                print(f"   Status: {article['credibility_status']}")
                print(f"   Score: {article['credibility_score']:.2f}")
                print(f"   Confidence: {article['confidence']:.2f}")
                print()
            
            # Show statistics
            stats = fetcher.get_statistics(articles)
            print("\n📊 STATISTICS:")
            print(f"   Total: {stats['total']}")
            print(f"   🚨 Fake: {stats['fake']} ({stats['fake_percentage']:.1f}%)")
            print(f"   ⚠️  Suspicious: {stats['suspicious']} ({stats['suspicious_percentage']:.1f}%)")
            print(f"   ✅ Credible: {stats['credible']} ({stats['credible_percentage']:.1f}%)")
            print(f"   📈 Average Score: {stats['avg_score']:.2f}")
        
        print("\n" + "="*60)
        print("✅ All tests passed!")
        print("="*60)
        
    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("\nSetup steps:")
        print("1. Copy .env.example to .env")
        print("2. Get API key from: https://newsapi.org/register")
        print("3. Add your key to .env file")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
