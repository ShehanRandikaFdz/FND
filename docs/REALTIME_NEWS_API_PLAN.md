# Real-Time Fake News Detector - Implementation Plan

**Date:** October 12, 2025  
**Project:** Upgrade FND to Real-Time News Monitoring System  
**Status:** 📋 Planning Phase

---

## 🎯 Objective

Transform the current manual text input system into an **automated real-time fake news detection system** that continuously monitors international news sources and flags potentially false information.

---

## 📊 System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Real-Time FND System                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │     News API Integration Layer          │
        │  (NewsAPI, Guardian, Reuters, etc.)     │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │      News Aggregation Service           │
        │  - Fetch articles every N minutes       │
        │  - Deduplicate similar articles         │
        │  - Filter by keywords/categories        │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │    Credibility Analysis Pipeline        │
        │  - Existing SVM/LSTM/BERT models        │
        │  - Batch processing for efficiency      │
        │  - Priority queue for breaking news     │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │        Alert & Storage System           │
        │  - Database (SQLite/PostgreSQL)         │
        │  - Real-time dashboard                  │
        │  - Email/SMS alerts for fake news       │
        └─────────────────────────────────────────┘
```

---

## 🌐 Recommended News APIs

### **Tier 1: Free APIs (Start Here)**

#### 1. **NewsAPI.org** ⭐ RECOMMENDED
- **URL:** https://newsapi.org
- **Free Tier:** 100 requests/day, 1000 articles/day
- **Coverage:** 80,000+ sources worldwide
- **Features:**
  - Everything endpoint (all articles)
  - Top headlines endpoint
  - Search by keyword, category, country
  - Historical data (up to 1 month)
- **Cost:** $0 (Free) → $449/month (Business)
- **Best For:** Getting started, proof of concept

```python
# Example NewsAPI Usage
from newsapi import NewsApiClient
newsapi = NewsApiClient(api_key='YOUR_API_KEY')

# Get top headlines
headlines = newsapi.get_top_headlines(
    language='en',
    country='us',
    page_size=100
)

# Search everything
all_articles = newsapi.get_everything(
    q='climate change',
    language='en',
    sort_by='publishedAt',
    page_size=100
)
```

#### 2. **The Guardian API** ⭐ RECOMMENDED
- **URL:** https://open-platform.theguardian.com
- **Free Tier:** 5,000 requests/day
- **Coverage:** The Guardian's content archive
- **Features:**
  - Full article text
  - Metadata and tags
  - No commercial restrictions
- **Cost:** 100% Free
- **Best For:** Quality journalism, UK/International news

```python
# Example Guardian API Usage
import requests

API_KEY = 'your-api-key'
url = f'https://content.guardianapis.com/search?api-key={API_KEY}'
params = {
    'show-fields': 'headline,body,byline',
    'page-size': 50,
    'order-by': 'newest'
}
response = requests.get(url, params=params)
```

#### 3. **GNews API**
- **URL:** https://gnews.io
- **Free Tier:** 100 requests/day
- **Coverage:** 60,000+ sources
- **Features:** Real-time news, full articles
- **Cost:** $0 (Free) → $29/month (Starter)

#### 4. **Currents API**
- **URL:** https://currentsapi.services
- **Free Tier:** 600 requests/day
- **Coverage:** News from 90+ countries
- **Cost:** Free tier available

### **Tier 2: Premium APIs (Scale Later)**

#### 5. **Reuters Connect API**
- **Coverage:** Global news, high credibility
- **Cost:** Enterprise pricing (contact sales)
- **Best For:** Financial news, breaking news

#### 6. **Associated Press (AP) API**
- **Coverage:** Global coverage, highly trusted
- **Cost:** Enterprise pricing
- **Best For:** Verified news sources

#### 7. **Bing News Search API** (Microsoft Azure)
- **Free Tier:** 1,000 transactions/month
- **Coverage:** Global aggregated news
- **Cost:** $3-7 per 1,000 transactions

### **Tier 3: Specialized Sources**

#### 8. **MediaStack**
- **Free Tier:** 500 requests/month
- **Coverage:** 7,500+ sources
- **Cost:** $0 (Free) → $49/month

#### 9. **New York Times API**
- **Free Tier:** 4,000 requests/day
- **Coverage:** NYT articles only
- **Cost:** Free for non-commercial

#### 10. **RSS Feeds** (100% Free Alternative)
- BBC News: http://feeds.bbci.co.uk/news/rss.xml
- CNN: http://rss.cnn.com/rss/edition.rss
- Reuters: https://www.rte.ie/rss/news.xml
- Al Jazeera: https://www.aljazeera.com/xml/rss/all.xml

---

## 🏗️ Implementation Phases

### **Phase 1: Foundation (Week 1-2)** 🟢 PRIORITY

#### 1.1 API Integration
- [ ] Register for NewsAPI.org (free tier)
- [ ] Register for The Guardian API (free tier)
- [ ] Create `news_api_client.py` module
- [ ] Implement rate limiting and error handling
- [ ] Test API connectivity

#### 1.2 Basic News Fetching
- [ ] Create `news_fetcher.py` service
- [ ] Implement article fetching every 30 minutes
- [ ] Add deduplication logic (by URL, title similarity)
- [ ] Store raw articles in SQLite database

**Deliverable:** Basic news fetching system running locally

**Estimated Time:** 5-7 days

---

### **Phase 2: Real-Time Processing (Week 3-4)** 🟡

#### 2.1 Processing Pipeline
- [ ] Create `realtime_processor.py`
- [ ] Implement background task scheduler (APScheduler)
- [ ] Batch process articles through existing models
- [ ] Optimize for performance (parallel processing)

#### 2.2 Database Design
```sql
-- Articles Table
CREATE TABLE articles (
    id INTEGER PRIMARY KEY,
    url TEXT UNIQUE,
    title TEXT,
    content TEXT,
    source TEXT,
    published_at DATETIME,
    fetched_at DATETIME,
    processed BOOLEAN DEFAULT FALSE
);

-- Analysis Results Table
CREATE TABLE analysis_results (
    id INTEGER PRIMARY KEY,
    article_id INTEGER,
    credibility_score REAL,
    credibility_status TEXT,
    confidence REAL,
    svm_prediction TEXT,
    lstm_prediction TEXT,
    bert_prediction TEXT,
    analyzed_at DATETIME,
    FOREIGN KEY (article_id) REFERENCES articles(id)
);

-- Alerts Table
CREATE TABLE alerts (
    id INTEGER PRIMARY KEY,
    article_id INTEGER,
    alert_type TEXT,
    severity TEXT,
    sent_at DATETIME,
    FOREIGN KEY (article_id) REFERENCES articles(id)
);
```

#### 2.3 Alert System
- [ ] Email alerts for fake news (threshold < 0.4)
- [ ] Telegram bot notifications
- [ ] Real-time dashboard updates

**Deliverable:** Automated processing pipeline with alerts

**Estimated Time:** 7-10 days

---

### **Phase 3: Advanced Features (Week 5-6)** 🟠

#### 3.1 Enhanced Dashboard
- [ ] Live news feed with credibility scores
- [ ] Filter by source, category, credibility
- [ ] Trending fake news topics
- [ ] Historical analysis charts
- [ ] Export reports (PDF, CSV)

#### 3.2 Smart Filtering
- [ ] Keyword-based monitoring (user-defined topics)
- [ ] Priority queue for breaking news
- [ ] Source reputation scoring
- [ ] Cross-reference with fact-checking sites

#### 3.3 Multi-Source Verification
- [ ] Compare same story across multiple sources
- [ ] Detect inconsistencies
- [ ] Calculate consensus score

**Deliverable:** Production-ready real-time monitoring system

**Estimated Time:** 10-14 days

---

### **Phase 4: Scaling & Optimization (Week 7-8)** 🔴

#### 4.1 Performance Optimization
- [ ] Redis caching for API responses
- [ ] Celery for distributed task processing
- [ ] PostgreSQL for production database
- [ ] Load balancing for multiple workers

#### 4.2 API Expansion
- [ ] Integrate 3-5 additional news APIs
- [ ] RSS feed parser for free sources
- [ ] Web scraping fallback (BeautifulSoup)

#### 4.3 Monitoring & Logging
- [ ] System health dashboard
- [ ] Error tracking (Sentry)
- [ ] Performance metrics
- [ ] API usage monitoring

**Deliverable:** Enterprise-grade scalable system

**Estimated Time:** 10-14 days

---

## 💻 Technical Stack

### Core Components

```yaml
Backend:
  - Python 3.13
  - FastAPI (REST API server)
  - APScheduler (task scheduling)
  - Celery (distributed tasks) [Optional]
  
APIs & Data:
  - NewsAPI (primary)
  - The Guardian API (secondary)
  - feedparser (RSS feeds)
  - requests (HTTP client)
  
Database:
  - SQLite (development)
  - PostgreSQL (production)
  - Redis (caching)
  
ML Models:
  - Existing SVM/LSTM/BERT models
  - No retraining needed initially
  
Frontend:
  - Streamlit (quick dashboard)
  - OR React (production dashboard)
  
Deployment:
  - Docker (containerization)
  - AWS EC2 / Google Cloud Run
  - GitHub Actions (CI/CD)
```

---

## 📁 New Project Structure

```
FND/
├── app.py                          # Existing Streamlit app
├── api_server.py                   # 🆕 FastAPI server
├── requirements.txt                # Update with new deps
├── config.py                       # 🆕 Configuration management
│
├── news_apis/                      # 🆕 API Integration
│   ├── __init__.py
│   ├── newsapi_client.py          # NewsAPI integration
│   ├── guardian_client.py         # Guardian API integration
│   ├── rss_parser.py              # RSS feed parser
│   └── base_client.py             # Base API client class
│
├── realtime/                       # 🆕 Real-time processing
│   ├── __init__.py
│   ├── news_fetcher.py            # Fetch news periodically
│   ├── processor.py               # Process articles
│   ├── scheduler.py               # Task scheduling
│   ├── deduplicator.py            # Remove duplicates
│   └── alert_system.py            # Send alerts
│
├── database/                       # 🆕 Database management
│   ├── __init__.py
│   ├── models.py                  # SQLAlchemy models
│   ├── db_manager.py              # Database operations
│   └── migrations/                # Database migrations
│
├── dashboard/                      # 🆕 Real-time dashboard
│   ├── __init__.py
│   ├── realtime_app.py            # New Streamlit dashboard
│   └── components/                # Dashboard components
│
├── credibility_analyzer/           # Existing (no changes)
├── utils/                          # Existing (add new utilities)
├── verdict_agent/                  # Existing (no changes)
├── models/                         # Existing (no changes)
├── tests/                          # Add new tests
└── docs/                           # Add API documentation
```

---

## 🔧 Implementation Code Templates

### 1. News API Client (`news_apis/newsapi_client.py`)

```python
"""
NewsAPI.org Integration Client
"""
import os
import requests
import time
from typing import List, Dict, Optional
from datetime import datetime, timedelta

class NewsAPIClient:
    """Client for NewsAPI.org integration"""
    
    BASE_URL = "https://newsapi.org/v2"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('NEWSAPI_KEY')
        self.session = requests.Session()
        self.rate_limit_remaining = 100
        self.rate_limit_reset = None
        
    def get_top_headlines(
        self,
        country: str = 'us',
        category: Optional[str] = None,
        page_size: int = 100
    ) -> List[Dict]:
        """Fetch top headlines"""
        endpoint = f"{self.BASE_URL}/top-headlines"
        params = {
            'apiKey': self.api_key,
            'country': country,
            'pageSize': page_size
        }
        if category:
            params['category'] = category
            
        return self._make_request(endpoint, params)
    
    def search_everything(
        self,
        query: Optional[str] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        language: str = 'en',
        sort_by: str = 'publishedAt',
        page_size: int = 100
    ) -> List[Dict]:
        """Search all articles"""
        endpoint = f"{self.BASE_URL}/everything"
        params = {
            'apiKey': self.api_key,
            'language': language,
            'sortBy': sort_by,
            'pageSize': page_size
        }
        
        if query:
            params['q'] = query
        if from_date:
            params['from'] = from_date.isoformat()
        if to_date:
            params['to'] = to_date.isoformat()
            
        return self._make_request(endpoint, params)
    
    def _make_request(self, endpoint: str, params: Dict) -> List[Dict]:
        """Make API request with rate limiting"""
        # Check rate limit
        if self.rate_limit_remaining == 0:
            if self.rate_limit_reset:
                wait_time = (self.rate_limit_reset - time.time())
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
                return data.get('articles', [])
            else:
                print(f"❌ API Error: {data.get('message', 'Unknown error')}")
                return []
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
            return []
    
    def get_sources(self, language: str = 'en') -> List[Dict]:
        """Get available news sources"""
        endpoint = f"{self.BASE_URL}/sources"
        params = {
            'apiKey': self.api_key,
            'language': language
        }
        
        try:
            response = self.session.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data.get('sources', [])
        except Exception as e:
            print(f"❌ Failed to fetch sources: {e}")
            return []


# Example usage
if __name__ == "__main__":
    client = NewsAPIClient()
    
    # Fetch top headlines
    print("📰 Fetching top headlines...")
    headlines = client.get_top_headlines(country='us', page_size=10)
    
    for article in headlines[:5]:
        print(f"\n📌 {article['title']}")
        print(f"   Source: {article['source']['name']}")
        print(f"   Published: {article['publishedAt']}")
        print(f"   URL: {article['url']}")
```

### 2. News Fetcher Service (`realtime/news_fetcher.py`)

```python
"""
Real-Time News Fetcher Service
Periodically fetches news from multiple sources
"""
import time
from datetime import datetime, timedelta
from typing import List, Dict
import hashlib
from difflib import SequenceMatcher

from news_apis.newsapi_client import NewsAPIClient
from news_apis.guardian_client import GuardianAPIClient
from database.db_manager import DatabaseManager

class NewsFetcher:
    """Fetches and manages news articles from multiple sources"""
    
    def __init__(self):
        self.newsapi = NewsAPIClient()
        self.guardian = GuardianAPIClient()
        self.db = DatabaseManager()
        self.seen_urls = set()
        self.seen_hashes = set()
        
    def fetch_all_sources(self) -> List[Dict]:
        """Fetch from all configured news sources"""
        all_articles = []
        
        # NewsAPI
        print("📡 Fetching from NewsAPI...")
        newsapi_articles = self.newsapi.get_top_headlines(page_size=100)
        all_articles.extend(self._normalize_newsapi(newsapi_articles))
        
        # The Guardian
        print("📡 Fetching from The Guardian...")
        guardian_articles = self.guardian.get_recent_articles(page_size=50)
        all_articles.extend(self._normalize_guardian(guardian_articles))
        
        # Deduplicate
        unique_articles = self._deduplicate(all_articles)
        
        print(f"✅ Fetched {len(all_articles)} articles, {len(unique_articles)} unique")
        return unique_articles
    
    def _normalize_newsapi(self, articles: List[Dict]) -> List[Dict]:
        """Normalize NewsAPI format"""
        normalized = []
        for article in articles:
            normalized.append({
                'url': article.get('url'),
                'title': article.get('title'),
                'content': article.get('content') or article.get('description', ''),
                'source': article['source']['name'],
                'published_at': article.get('publishedAt'),
                'author': article.get('author'),
                'image_url': article.get('urlToImage'),
                'api_source': 'newsapi'
            })
        return normalized
    
    def _normalize_guardian(self, articles: List[Dict]) -> List[Dict]:
        """Normalize Guardian API format"""
        normalized = []
        for article in articles:
            normalized.append({
                'url': article.get('webUrl'),
                'title': article.get('webTitle'),
                'content': article.get('fields', {}).get('body', ''),
                'source': 'The Guardian',
                'published_at': article.get('webPublicationDate'),
                'author': article.get('fields', {}).get('byline'),
                'image_url': article.get('fields', {}).get('thumbnail'),
                'api_source': 'guardian'
            })
        return normalized
    
    def _deduplicate(self, articles: List[Dict]) -> List[Dict]:
        """Remove duplicate articles"""
        unique = []
        
        for article in articles:
            url = article.get('url')
            title = article.get('title', '')
            
            # Skip if URL already seen
            if url in self.seen_urls:
                continue
            
            # Generate content hash
            content_hash = self._get_content_hash(title, article.get('content', ''))
            
            # Skip if similar content already seen
            if content_hash in self.seen_hashes:
                continue
            
            # Check title similarity with existing articles
            is_duplicate = False
            for existing in unique:
                similarity = self._title_similarity(
                    title, 
                    existing.get('title', '')
                )
                if similarity > 0.85:  # 85% similar
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(article)
                self.seen_urls.add(url)
                self.seen_hashes.add(content_hash)
        
        return unique
    
    def _get_content_hash(self, title: str, content: str) -> str:
        """Generate hash for content deduplication"""
        text = f"{title} {content}".lower()
        return hashlib.md5(text.encode()).hexdigest()
    
    def _title_similarity(self, title1: str, title2: str) -> float:
        """Calculate similarity between two titles"""
        return SequenceMatcher(None, title1.lower(), title2.lower()).ratio()
    
    def save_articles(self, articles: List[Dict]) -> int:
        """Save articles to database"""
        saved_count = 0
        for article in articles:
            if self.db.insert_article(article):
                saved_count += 1
        return saved_count


# Example usage
if __name__ == "__main__":
    fetcher = NewsFetcher()
    
    print("🚀 Starting news fetch...")
    articles = fetcher.fetch_all_sources()
    
    print(f"\n💾 Saving to database...")
    saved = fetcher.save_articles(articles)
    print(f"✅ Saved {saved} new articles")
```

### 3. Real-Time Processor (`realtime/processor.py`)

```python
"""
Real-Time News Processing Pipeline
Analyzes fetched articles for credibility
"""
import time
from typing import List, Dict
from datetime import datetime

from credibility_analyzer.credibility_analyzer import CredibilityAnalyzer
from database.db_manager import DatabaseManager
from realtime.alert_system import AlertSystem

class RealtimeProcessor:
    """Process news articles through credibility analysis"""
    
    def __init__(self):
        print("🔍 Loading Credibility Analyzer...")
        self.analyzer = CredibilityAnalyzer()
        self.db = DatabaseManager()
        self.alert_system = AlertSystem()
        
        # Thresholds
        self.fake_threshold = 0.4  # Below this = likely fake
        self.suspicious_threshold = 0.6  # Between = suspicious
        
    def process_pending_articles(self):
        """Process all unprocessed articles"""
        pending = self.db.get_unprocessed_articles()
        
        if not pending:
            print("✅ No pending articles to process")
            return
        
        print(f"📊 Processing {len(pending)} articles...")
        
        for i, article in enumerate(pending, 1):
            print(f"\n[{i}/{len(pending)}] Processing: {article['title'][:60]}...")
            
            result = self._analyze_article(article)
            
            if result:
                self.db.save_analysis_result(article['id'], result)
                
                # Check if alert needed
                if result['credibility_score'] < self.fake_threshold:
                    self.alert_system.send_alert(article, result, severity='HIGH')
                elif result['credibility_score'] < self.suspicious_threshold:
                    self.alert_system.send_alert(article, result, severity='MEDIUM')
            
            # Rate limiting
            time.sleep(0.1)
        
        print(f"\n✅ Processed {len(pending)} articles")
    
    def _analyze_article(self, article: Dict) -> Dict:
        """Analyze single article"""
        try:
            # Combine title and content
            text = f"{article['title']}. {article['content']}"
            
            # Analyze
            result = self.analyzer.analyze_credibility(text)
            
            # Add metadata
            result['analyzed_at'] = datetime.now()
            result['article_id'] = article['id']
            
            return result
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            return None
    
    def process_single_article(self, article: Dict) -> Dict:
        """Process a single article (for testing)"""
        return self._analyze_article(article)


# Example usage
if __name__ == "__main__":
    processor = RealtimeProcessor()
    
    print("🚀 Starting real-time processing...")
    processor.process_pending_articles()
```

### 4. Task Scheduler (`realtime/scheduler.py`)

```python
"""
Background Task Scheduler
Runs news fetching and processing periodically
"""
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime
import signal
import sys

from realtime.news_fetcher import NewsFetcher
from realtime.processor import RealtimeProcessor

class NewsScheduler:
    """Schedule periodic news fetching and processing"""
    
    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.fetcher = NewsFetcher()
        self.processor = RealtimeProcessor()
        
        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)
    
    def start(self):
        """Start the scheduler"""
        print("🚀 Starting News Monitoring Service...")
        print("=" * 60)
        
        # Fetch news every 30 minutes
        self.scheduler.add_job(
            func=self._fetch_news,
            trigger=IntervalTrigger(minutes=30),
            id='fetch_news',
            name='Fetch news from APIs',
            replace_existing=True
        )
        
        # Process articles every 5 minutes
        self.scheduler.add_job(
            func=self._process_news,
            trigger=IntervalTrigger(minutes=5),
            id='process_news',
            name='Process pending articles',
            replace_existing=True
        )
        
        # Cleanup old data daily
        self.scheduler.add_job(
            func=self._cleanup_old_data,
            trigger='cron',
            hour=2,  # 2 AM
            id='cleanup',
            name='Daily cleanup',
            replace_existing=True
        )
        
        # Start scheduler
        self.scheduler.start()
        
        # Run initial fetch immediately
        print("\n🔄 Running initial fetch...")
        self._fetch_news()
        self._process_news()
        
        print("\n✅ Scheduler started successfully!")
        print("📊 Active jobs:")
        for job in self.scheduler.get_jobs():
            print(f"   - {job.name} (next run: {job.next_run_time})")
        print("\n⏳ Press Ctrl+C to stop...\n")
        
        # Keep running
        try:
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            self._shutdown()
    
    def _fetch_news(self):
        """Fetch news job"""
        print(f"\n{'='*60}")
        print(f"📡 FETCHING NEWS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        try:
            articles = self.fetcher.fetch_all_sources()
            saved = self.fetcher.save_articles(articles)
            print(f"✅ Fetch complete: {saved} new articles saved")
        except Exception as e:
            print(f"❌ Fetch failed: {e}")
    
    def _process_news(self):
        """Process news job"""
        print(f"\n{'='*60}")
        print(f"🔍 PROCESSING NEWS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        try:
            self.processor.process_pending_articles()
            print(f"✅ Processing complete")
        except Exception as e:
            print(f"❌ Processing failed: {e}")
    
    def _cleanup_old_data(self):
        """Cleanup old data job"""
        print(f"\n🧹 Running daily cleanup...")
        # Implement cleanup logic (e.g., delete articles older than 30 days)
        pass
    
    def _shutdown(self, signum=None, frame=None):
        """Graceful shutdown"""
        print("\n\n🛑 Shutting down scheduler...")
        self.scheduler.shutdown()
        print("✅ Shutdown complete")
        sys.exit(0)


# Run scheduler
if __name__ == "__main__":
    scheduler = NewsScheduler()
    scheduler.start()
```

### 5. Configuration (`config.py`)

```python
"""
Configuration Management
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration"""
    
    # API Keys
    NEWSAPI_KEY = os.getenv('NEWSAPI_KEY', '')
    GUARDIAN_API_KEY = os.getenv('GUARDIAN_API_KEY', '')
    
    # Database
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///news_monitor.db')
    
    # Scheduling
    FETCH_INTERVAL_MINUTES = int(os.getenv('FETCH_INTERVAL_MINUTES', 30))
    PROCESS_INTERVAL_MINUTES = int(os.getenv('PROCESS_INTERVAL_MINUTES', 5))
    
    # Credibility Thresholds
    FAKE_THRESHOLD = float(os.getenv('FAKE_THRESHOLD', 0.4))
    SUSPICIOUS_THRESHOLD = float(os.getenv('SUSPICIOUS_THRESHOLD', 0.6))
    
    # Alerts
    ENABLE_EMAIL_ALERTS = os.getenv('ENABLE_EMAIL_ALERTS', 'false').lower() == 'true'
    EMAIL_RECIPIENT = os.getenv('EMAIL_RECIPIENT', '')
    
    ENABLE_TELEGRAM_ALERTS = os.getenv('ENABLE_TELEGRAM_ALERTS', 'false').lower() == 'true'
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
    
    # News Sources
    NEWS_COUNTRIES = os.getenv('NEWS_COUNTRIES', 'us,gb,ca').split(',')
    NEWS_CATEGORIES = os.getenv('NEWS_CATEGORIES', 'general,business,technology').split(',')
```

---

## 📦 Required Dependencies

Add to `requirements.txt`:

```txt
# Existing dependencies
streamlit==1.50.0
tensorflow==2.20.0
torch==2.8.0
transformers==4.56.2
scikit-learn==1.6.1
pandas==2.2.3
numpy==2.3.0

# NEW: API Integration
newsapi-python==0.2.7
requests==2.32.3
feedparser==6.0.11
python-dotenv==1.0.1

# NEW: Database
SQLAlchemy==2.0.36
alembic==1.14.0
psycopg2-binary==2.9.10  # PostgreSQL (optional)

# NEW: Task Scheduling
APScheduler==3.10.4

# NEW: Web Server (optional)
fastapi==0.115.6
uvicorn==0.34.0

# NEW: Alerts
python-telegram-bot==20.7

# NEW: Utilities
python-dateutil==2.9.0
pytz==2024.2
```

---

## 🎯 Quick Start Guide

### Step 1: Get API Keys (5 minutes)

```bash
# 1. NewsAPI.org
# Visit: https://newsapi.org/register
# Copy your API key

# 2. The Guardian
# Visit: https://bonobo.capi.gutools.co.uk/register/developer
# Copy your API key
```

### Step 2: Setup Environment

```powershell
# Create .env file
New-Item -Path .env -ItemType File

# Add API keys
@"
NEWSAPI_KEY=your_newsapi_key_here
GUARDIAN_API_KEY=your_guardian_key_here
DATABASE_URL=sqlite:///news_monitor.db
FETCH_INTERVAL_MINUTES=30
PROCESS_INTERVAL_MINUTES=5
"@ | Out-File -FilePath .env
```

### Step 3: Install Dependencies

```powershell
pip install newsapi-python requests feedparser python-dotenv SQLAlchemy APScheduler
```

### Step 4: Test API Connection

```python
# test_apis.py
from news_apis.newsapi_client import NewsAPIClient

client = NewsAPIClient()
headlines = client.get_top_headlines(page_size=5)

for article in headlines:
    print(f"✅ {article['title']}")
```

### Step 5: Run Initial Fetch

```powershell
python realtime/news_fetcher.py
```

### Step 6: Start Real-Time Monitoring

```powershell
python realtime/scheduler.py
```

---

## 📊 Success Metrics

### Technical Metrics
- ✅ Fetch 500+ articles per hour
- ✅ Process articles within 5 minutes of fetching
- ✅ 99% uptime for monitoring service
- ✅ < 2% false positive rate on fake news detection

### Business Metrics
- ✅ Detect fake news within 30 minutes of publication
- ✅ Cover 50+ international news sources
- ✅ Process 5,000+ articles daily
- ✅ Send < 10 alerts per day (high-confidence fakes only)

---

## 💰 Cost Estimate

### Free Tier (0-3 months)
- NewsAPI: Free (100 req/day)
- Guardian API: Free (5000 req/day)
- Database: SQLite (free)
- Hosting: Local/Free tier cloud
- **Total: $0/month**

### Starter Tier (3-6 months)
- NewsAPI: $449/month (business)
- Guardian API: Free
- PostgreSQL: $15/month
- Cloud hosting: $20/month
- **Total: $484/month**

### Production Tier (6+ months)
- Multiple premium APIs: $500-1000/month
- PostgreSQL + Redis: $50/month
- Cloud hosting: $100/month
- Monitoring tools: $50/month
- **Total: $700-1200/month**

---

## ⚠️ Challenges & Solutions

### Challenge 1: Rate Limits
**Problem:** Free APIs have strict rate limits  
**Solution:**
- Use multiple API keys (rotate)
- Implement intelligent caching (Redis)
- Prioritize important sources
- Add RSS feed fallback

### Challenge 2: API Costs
**Problem:** Scaling requires expensive paid plans  
**Solution:**
- Start with free tier (proof of concept)
- Use RSS feeds for supplementary data
- Negotiate enterprise pricing when scaling
- Consider web scraping (legal compliance required)

### Challenge 3: Duplicate Content
**Problem:** Same story from multiple sources  
**Solution:**
- Implement robust deduplication (URL, title, content hash)
- Use NLP for semantic similarity detection
- Group related articles together

### Challenge 4: Processing Speed
**Problem:** 3 models = slow processing  
**Solution:**
- Batch processing (process 10-20 articles together)
- Parallel processing with multiprocessing
- Use GPU for BERT model
- Consider lighter models for real-time use

### Challenge 5: False Positives
**Problem:** Flagging real news as fake  
**Solution:**
- Set higher thresholds for alerts
- Require consensus from 2/3 models
- Manual review queue for borderline cases
- Continuous model retraining

---

## 🚀 Next Steps

### Immediate Actions (This Week)
1. ✅ Register for NewsAPI.org and Guardian API
2. ✅ Create `news_apis/` folder structure
3. ✅ Implement basic API client
4. ✅ Test API connectivity
5. ✅ Design database schema

### Short Term (Next 2 Weeks)
6. ✅ Build news fetching service
7. ✅ Implement deduplication logic
8. ✅ Create processing pipeline
9. ✅ Setup basic scheduling
10. ✅ Test end-to-end flow

### Medium Term (Next Month)
11. ✅ Build real-time dashboard
12. ✅ Add alert system
13. ✅ Integrate 3+ news sources
14. ✅ Optimize performance
15. ✅ Deploy to cloud

---

## 📞 Support & Resources

### Documentation
- NewsAPI: https://newsapi.org/docs
- Guardian API: https://open-platform.theguardian.com/documentation/
- APScheduler: https://apscheduler.readthedocs.io/
- SQLAlchemy: https://docs.sqlalchemy.org/

### Alternative Approaches
1. **RSS-Only Solution** (100% Free)
   - Parse RSS feeds from major news sites
   - No API limits
   - Slower updates

2. **Web Scraping** (Free but complex)
   - Direct scraping of news websites
   - Legal considerations required
   - More maintenance

3. **Hybrid Approach** (Recommended)
   - Premium APIs for breaking news
   - RSS feeds for supplementary coverage
   - Best balance of cost and coverage

---

## ✅ Decision Matrix: Which Approach?

| Approach | Cost | Speed | Coverage | Reliability | Recommendation |
|----------|------|-------|----------|-------------|----------------|
| **NewsAPI + Guardian** | $0-449/mo | Fast | Excellent | High | ⭐ **START HERE** |
| **RSS Feeds Only** | $0 | Medium | Good | Medium | Budget option |
| **Premium APIs Only** | $500+/mo | Very Fast | Excellent | Very High | Enterprise only |
| **Web Scraping** | $0 | Slow | Variable | Low | Not recommended |
| **Hybrid (API + RSS)** | $0-100/mo | Fast | Excellent | High | ⭐ **SCALE HERE** |

---

## 🎯 Recommended Path

```
Phase 1 (Week 1-2): NewsAPI Free Tier
└── Proof of concept, basic functionality

Phase 2 (Week 3-4): Add Guardian API + RSS
└── Expand coverage, test deduplication

Phase 3 (Month 2): Optimize & Scale
└── Performance tuning, better alerts

Phase 4 (Month 3+): Production Ready
└── Cloud deployment, premium APIs, monitoring
```

---

**Status:** 📋 Ready to implement  
**Estimated Timeline:** 6-8 weeks to production  
**Recommended Budget:** $0 (first 3 months) → $500/month (scaled)

---

Would you like me to start implementing Phase 1 (API integration) now?
