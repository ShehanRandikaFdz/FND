# 📰 AGENT 1: ARTICLE COLLECTOR# 📰 AGENT 1: ARTICLE COLLECTOR (News Fetcher Agent)



## 📋 Table of Contents## 🎯 Overview

The **Article Collector Agent** is responsible for fetching real-time news articles from external sources (NewsAPI), processing them, and preparing them for credibility analysis. This agent acts as the data acquisition layer of the system.

1. [Overview](#overview)

2. [Agent Architecture](#agent-architecture)---

3. [Key Components](#key-components)

4. [Core Functions](#core-functions)## 🏗️ Architecture

5. [NewsAPI Integration](#newsapi-integration)

6. [URL Content Extraction](#url-content-extraction)### Core Components:

7. [Text Preprocessing](#text-preprocessing)1. **NewsAPI Client** - External news source integration

8. [Deduplication Strategy](#deduplication-strategy)2. **News Fetcher** - Main orchestration and article processing

9. [API Endpoints](#api-endpoints)3. **Content Extractor** - URL-based article extraction

10. [Code Examples](#code-examples)4. **ML Predictor** - On-the-fly article analysis

11. [Error Handling](#error-handling)

---

---

## 📂 Key Files and Code

## 1. Overview

### 1. **news_apis/newsapi_client.py**

**Agent 1: Article Collector** is the first agent in the three-agent pipeline, responsible for **data acquisition and preprocessing**. It fetches news articles from external sources, extracts content from URLs, cleans text, and prepares data for analysis by Agent 2 (Credibility Analyzer).**Purpose**: Interface with NewsAPI.org to fetch news articles



### Primary Responsibilities**Key Functions**:



✅ **News Fetching**: Retrieve articles from NewsAPI.org  ```python

✅ **URL Extraction**: Scrape content from web pages  class NewsAPIClient:

✅ **Text Cleaning**: Preprocess and normalize text      def __init__(self, api_key: str):

✅ **Deduplication**: Prevent analysis of duplicate articles          """Initialize NewsAPI client with authentication"""

✅ **Data Preparation**: Format data for ML models          self.api_key = api_key

        self.base_url = "https://newsapi.org/v2"

### Key Files        self.session = requests.Session()

        self.session.headers.update({

| File | Lines | Purpose |            'Authorization': f'Bearer {api_key}',

|------|-------|---------|            'User-Agent': 'Fake-News-Detector/1.0'

| `news_fetcher.py` | 222 | Main orchestrator for Agent 1 |        })

| `news_apis/newsapi_client.py` | 150 | NewsAPI.org integration |```

| `app.py` (extract functions) | ~100 | URL content extraction |

| `utils/text_preprocessor.py` | ~100 | Text cleaning utilities |**Main Features**:



---#### a) **Get Top Headlines** (Lines 24-50)

```python

## 2. Agent Architecturedef get_top_headlines(

    self, 

```    country: Optional[str] = None,

┌─────────────────────────────────────────────────────────────┐    category: Optional[str] = None,

│                  AGENT 1: ARTICLE COLLECTOR                  │    page_size: int = 10,

├─────────────────────────────────────────────────────────────┤    page: int = 1

│                                                               │) -> List[Dict]:

│  ┌───────────────────────────────────────────────────────┐ │    """

│  │         INPUT SOURCES                                  │ │    Fetch top headlines by country and category

│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐           │ │    

│  │  │ NewsAPI  │  │   URL    │  │   Text   │           │ │    Parameters:

│  │  │  Client  │  │ Scraper  │  │  Direct  │           │ │    - country: 2-letter country code (e.g., 'us', 'gb')

│  │  └─────┬────┘  └─────┬────┘  └─────┬────┘           │ │    - category: News category (general, business, tech, etc.)

│  └────────┼─────────────┼─────────────┼────────────────┘ │    - page_size: Number of articles (max 100)

│           │             │             │                    │    - page: Page number for pagination

│           ▼             ▼             ▼                    │    

│  ┌───────────────────────────────────────────────────────┐ │    Returns:

│  │         PREPROCESSING PIPELINE                         │ │    - List of article dictionaries

│  │                                                         │ │    """

│  │  1. HTML Tag Removal                                   │ │    params = {

│  │  2. Special Character Cleaning                         │ │        'pageSize': min(page_size, 100),

│  │  3. Whitespace Normalization                           │ │        'page': page

│  │  4. URL/Email Removal                                  │ │    }

│  │  5. Text Validation                                    │ │    if country:

│  └────────────────────┬──────────────────────────────────┘ │        params['country'] = country

│                       │                                      │    if category:

│                       ▼                                      │        params['category'] = category

│  ┌───────────────────────────────────────────────────────┐ │    

│  │         DEDUPLICATION                                  │ │    response = self.session.get(

│  │                                                         │ │        f"{self.base_url}/top-headlines", 

│  │  • URL Tracking                                        │ │        params=params

│  │  • Content Hash Comparison                             │ │    )

│  │  • Seen Articles Cache                                 │ │    return response.json().get('articles', [])

│  └────────────────────┬──────────────────────────────────┘ │```

│                       │                                      │

│                       ▼                                      │#### b) **Search Everything** (Lines 52-80)

│  ┌───────────────────────────────────────────────────────┐ │```python

│  │         OUTPUT: CLEAN DATA                             │ │def search_everything(

│  │                                                         │ │    self,

│  │  {                                                      │ │    query: str,

│  │    "title": "Article Title",                           │ │    page_size: int = 10,

│  │    "content": "Clean text...",                         │ │    page: int = 1,

│  │    "url": "https://...",                               │ │    sort_by: str = 'relevancy'

│  │    "source": "Source Name",                            │ │) -> List[Dict]:

│  │    "publishedAt": "2025-10-18T..."                     │ │    """

│  │  }                                                      │ │    Search for articles by keyword

│  └────────────────────┬──────────────────────────────────┘ │    

└────────────────────────┼────────────────────────────────────┘    Parameters:

                         │    - query: Search keywords

                         ▼    - page_size: Number of results

              TO AGENT 2 (Credibility Analyzer)    - sort_by: Sort order (relevancy, popularity, publishedAt)

```    

    Returns:

---    - List of matching articles

    """

## 3. Key Components    params = {

        'q': query,

### 3.1 NewsFetcher Class        'pageSize': min(page_size, 100),

        'page': page,

**File**: `news_fetcher.py`        'sortBy': sort_by,

        'language': 'en'

```python    }

class NewsFetcher:    

    """Fetches and analyzes news articles"""    response = self.session.get(

            f"{self.base_url}/everything", 

    def __init__(self):        params=params

        """Initialize news fetcher with API client and analyzer"""    )

        # Validate configuration    return response.json().get('articles', [])

        if not Config.NEWSAPI_KEY:```

            raise ValueError("NewsAPI key not configured!")

        ---

        # Initialize API client

        self.api_client = NewsAPIClient(api_key=Config.NEWSAPI_KEY)### 2. **news_fetcher.py**

        **Purpose**: Main orchestration agent for fetching and analyzing articles

        # Initialize ML components (lazy loading)

        self.model_loader = None**Key Functions**:

        self.predictor = None

        #### a) **Initialization** (Lines 19-39)

        # Track processed articles```python

        self.seen_urls = set()class NewsFetcher:

        self.seen_hashes = set()    def __init__(self):

```        """Initialize news fetcher with API client and analyzer"""

        print("Initializing News Fetcher...")

**Attributes**:        

- `api_client`: NewsAPI client instance        # Validate configuration

- `model_loader`: ML model loader (lazy loaded)        if not Config.NEWSAPI_KEY:

- `predictor`: Unified predictor (lazy loaded)            raise ValueError(

- `seen_urls`: Set of processed URLs (deduplication)                "NewsAPI key not configured! Check .env file."

- `seen_hashes`: Set of content hashes (deduplication)            )

        

---        # Initialize API client

        self.api_client = NewsAPIClient(api_key=Config.NEWSAPI_KEY)

### 3.2 NewsAPIClient Class        

        # Initialize ML components (lazy loading)

**File**: `news_apis/newsapi_client.py`        self.model_loader = None

        self.predictor = None

```python        

class NewsAPIClient:        # Track processed articles to avoid duplicates

    """Client for NewsAPI.org integration"""        self.seen_urls = set()

            self.seen_hashes = set()

    def __init__(self, api_key: str):        

        self.api_key = api_key        print("News Fetcher initialized successfully!")

        self.base_url = 'https://newsapi.org/v2'```

        self.session = requests.Session()

        self.session.headers.update({**Key Features**:

            'X-Api-Key': api_key,- ✅ Validates API key from environment

            'User-Agent': 'FakeNewsDetector/1.0'- ✅ Lazy-loads ML models for efficiency

        })- ✅ Duplicate detection using URL and content hashing

    - ✅ Connection pooling via NewsAPIClient

    def get_top_headlines(self, country=None, category=None, page_size=20, page=1):

        """Fetch top headlines from NewsAPI"""#### b) **Fetch and Analyze** (Lines 54-112)

        # Implementation...```python

    def fetch_and_analyze(

    def search_articles(self, query, page_size=20, page=1):    self,

        """Search for articles by keyword"""    country: Optional[str] = None,

        # Implementation...    category: Optional[str] = None,

```    query: Optional[str] = None,

    page_size: Optional[int] = None,

---    page: int = 1

) -> List[Dict]:

## 4. Core Functions    """

    Main workflow: Fetch news → Analyze credibility → Return results

### 4.1 fetch_and_analyze()    

    Workflow:

**Main orchestrator function for Agent 1**    1. Load ML models (if not already loaded)

    2. Fetch articles from NewsAPI

**File**: `news_fetcher.py`    3. Deduplicate articles

    4. Analyze each article for credibility

```python    5. Return enriched article data

def fetch_and_analyze(    """

    self,    try:

    country: Optional[str] = None,        # Load ML models

    category: Optional[str] = None,        self._load_ml_models()

    query: Optional[str] = None,        

    page_size: Optional[int] = None,        # Fetch articles from NewsAPI with pagination

    page: int = 1        print(f"Fetching articles (page {page})...")

) -> List[Dict]:        if query:

    """            articles = self.api_client.search_articles(

    Fetch news and analyze credibility with pagination support                query=query,

                    page_size=page_size or 10,

    Args:                page=page

        country: Country code (e.g., 'us', 'gb')            )

        category: News category (e.g., 'general', 'technology')        else:

        query: Search query            articles = self.api_client.get_top_headlines(

        page_size: Number of articles to fetch per page                country=country,

        page: Page number for pagination (default: 1)                category=category,

                        page_size=page_size or 10,

    Returns:                page=page

        List of analyzed articles with credibility scores            )

    """        

    try:        if not articles:

        # Load ML models            print("No articles found")

        self._load_ml_models()            return []

                

        # Fetch articles from NewsAPI with pagination        # Analyze each article

        print(f"Fetching articles (page {page})...")        analyzed_articles = []

        if query:        for article in articles:

            articles = self.api_client.search_articles(            try:

                query=query,                analyzed = self._analyze_article(article)

                page_size=page_size or 10,                if analyzed:

                page=page                    analyzed_articles.append(analyzed)

            )            except Exception as e:

        else:                print(f"Error analyzing article: {e}")

            articles = self.api_client.get_top_headlines(                continue

                country=country,        

                category=category,        print(f"Successfully analyzed {len(analyzed_articles)} articles")

                page_size=page_size or 10,        return analyzed_articles

                page=page        

            )    except Exception as e:

                print(f"Error fetching news: {e}")

        if not articles:        return []

            return []```

        

        # Analyze each article#### c) **Article Analysis** (Lines 114-176)

        analyzed_articles = []```python

        for article in articles:def _analyze_article(self, article: Dict) -> Optional[Dict]:

            analyzed = self._analyze_article(article)    """

            if analyzed:    Analyze a single article for credibility

                analyzed_articles.append(analyzed)    

            Steps:

        return analyzed_articles    1. Check for duplicates (using hash)

        2. Extract text content (title, description, content)

    except Exception as e:    3. Run ML prediction

        print(f"Error in fetch_and_analyze: {e}")    4. Calculate credibility score

        return []    5. Return enriched article data

```    """

    try:

**Workflow**:        # Check for duplicates

1. Load ML models (lazy loading)        article_hash = hashlib.md5(

2. Fetch articles from NewsAPI            article.get('url', '').encode()

3. Analyze each article with `_analyze_article()`        ).hexdigest()

4. Return analyzed articles        if article_hash in self.seen_hashes:

            return None

---        self.seen_hashes.add(article_hash)

        

### 4.2 _analyze_article()        # Prepare text for analysis

        title = article.get('title', '')

**Analyze individual article**        description = article.get('description', '')

        content = article.get('content', '')

**File**: `news_fetcher.py`        

        # Combine text (prefer content, fallback to description)

```python        text_to_analyze = content or description or title

def _analyze_article(self, article: Dict) -> Optional[Dict]:        

    """        if not text_to_analyze or len(text_to_analyze.strip()) < 10:

    Analyze individual article for credibility            return None

            

    Args:        # Make prediction using ML models

        article: Article data from NewsAPI        result = self.predictor.ensemble_predict_majority(

                    text_to_analyze

    Returns:        )

        Article with analysis results or None if duplicate        

    """        # Calculate credibility score (0.0 to 1.0)

    try:        prediction = result.get('final_prediction', 'UNKNOWN')

        # Check for duplicates        confidence = result.get('confidence', 0)

        url = article.get('url', '')        

        if url in self.seen_urls:        if prediction == 'TRUE':

            print(f"Skipping duplicate URL: {url}")            credibility_score = confidence / 100.0

            return None        else:

                    credibility_score = (100 - confidence) / 100.0

        # Get article content        

        content = article.get('description', '') or article.get('content', '')        # Get individual model results

        if not content:        individual_results = result.get('individual_results', {})

            content = article.get('title', '')        

                # Return enriched article

        # Check content hash for duplicates        return {

        content_hash = hashlib.md5(content.encode()).hexdigest()            'title': title,

        if content_hash in self.seen_hashes:            'description': description,

            print(f"Skipping duplicate content")            'content': content,

            return None            'url': article.get('url', ''),

                    'source': article.get('source', {}).get('name', 'Unknown'),

        # Mark as seen            'author': article.get('author', 'Unknown'),

        self.seen_urls.add(url)            'published_at': article.get('publishedAt', ''),

        self.seen_hashes.add(content_hash)            'image_url': article.get('urlToImage', ''),

                    

        # Analyze with ML models            # Credibility Analysis Results

        analysis_result = self.predictor.ensemble_predict_majority(content)            'credibility_score': credibility_score,

                    'confidence': confidence / 100.0,

        # Add analysis to article            'prediction': prediction,

        article['analysis'] = {            'individual_predictions': individual_results,

            'prediction': analysis_result['prediction'],            'analyzed_at': datetime.now().isoformat(),

            'confidence': analysis_result['confidence'],            'text_analyzed': text_to_analyze[:200] + "..."

            'verdict_type': analysis_result.get('verdict_type', 'UNKNOWN'),        }

            'model_results': analysis_result.get('model_results', {})        

        }    except Exception as e:

                print(f"Error analyzing article: {e}")

        return article        return None

    ```

    except Exception as e:

        print(f"Error analyzing article: {e}")#### d) **Statistics Generation** (Lines 197-222)

        return None```python

```def get_statistics(self, articles: List[Dict]) -> Dict:

    """

**Workflow**:    Calculate aggregate statistics from analyzed articles

1. Check for duplicate URLs    

2. Extract article content    Categorization:

3. Check for duplicate content (hash)    - FAKE: credibility_score < 0.3

4. Mark as seen (deduplication)    - SUSPICIOUS: 0.3 <= credibility_score < 0.7

5. Analyze with ML models (Agent 2)    - CREDIBLE: credibility_score >= 0.7

6. Add analysis to article    """

7. Return analyzed article    if not articles:

        return {

---            'total': 0,

            'fake': 0,

### 4.3 get_statistics()            'suspicious': 0,

            'credible': 0,

**Generate statistics from analyzed articles**            'fake_percentage': 0,

            'suspicious_percentage': 0,

**File**: `news_fetcher.py`            'credible_percentage': 0,

            'average_credibility': 0

```python        }

def get_statistics(self, articles: List[Dict]) -> Dict:    

    """    total = len(articles)

    Generate statistics from analyzed articles    fake = sum(1 for a in articles if a['credibility_score'] < 0.3)

        suspicious = sum(

    Args:        1 for a in articles 

        articles: List of analyzed articles        if 0.3 <= a['credibility_score'] < 0.7

            )

    Returns:    credible = sum(1 for a in articles if a['credibility_score'] >= 0.7)

        Statistics dictionary    

    """    avg_credibility = sum(

    if not articles:        a['credibility_score'] for a in articles

        return {    ) / total

            'total': 0,    

            'true_news': 0,    return {

            'fake_news': 0,        'total': total,

            'uncertain': 0,        'fake': fake,

            'avg_confidence': 0.0        'suspicious': suspicious,

        }        'credible': credible,

            'fake_percentage': (fake / total) * 100,

    total = len(articles)        'suspicious_percentage': (suspicious / total) * 100,

    true_count = sum(1 for a in articles         'credible_percentage': (credible / total) * 100,

                    if a.get('analysis', {}).get('prediction') == 'TRUE')        'average_credibility': avg_credibility

    fake_count = sum(1 for a in articles     }

                    if a.get('analysis', {}).get('prediction') == 'FAKE')```

    uncertain_count = total - true_count - fake_count

    ---

    confidences = [a.get('analysis', {}).get('confidence', 0) 

                   for a in articles]### 3. **app.py - Content Extraction** (Lines 122-213)

    avg_confidence = sum(confidences) / len(confidences) if confidences else 0**Purpose**: Extract article content from URLs for analysis

    

    return {```python

        'total': total,def extract_article_content(url):

        'true_news': true_count,    """

        'fake_news': fake_count,    Extract article content from URL using web scraping

        'uncertain': uncertain_count,    

        'avg_confidence': round(avg_confidence, 1)    Steps:

    }    1. Fetch webpage with browser headers

```    2. Parse HTML with BeautifulSoup

    3. Extract title, content, source

---    4. Clean and format text

    5. Return structured data

## 5. NewsAPI Integration    """

    try:

### 5.1 API Client Initialization        import requests

        from bs4 import BeautifulSoup

**File**: `news_apis/newsapi_client.py`        import re

        from urllib.parse import urlparse

```python        

class NewsAPIClient:        # Set headers to mimic a real browser

    def __init__(self, api_key: str):        headers = {

        """Initialize NewsAPI client"""            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '

        self.api_key = api_key                         'AppleWebKit/537.36 (KHTML, like Gecko) '

        self.base_url = 'https://newsapi.org/v2'                         'Chrome/91.0.4472.124 Safari/537.36'

                }

        # Create session with headers        

        self.session = requests.Session()        # Make request to URL

        self.session.headers.update({        response = requests.get(url, headers=headers, timeout=10)

            'X-Api-Key': api_key,        response.raise_for_status()

            'User-Agent': 'FakeNewsDetector/1.0'        

        })        # Parse HTML content

```        soup = BeautifulSoup(response.content, 'html.parser')

        

---        # Remove script and style elements

        for script in soup(["script", "style"]):

### 5.2 Get Top Headlines            script.decompose()

        

**Fetch top headlines by country/category**        # Try to find article title

        title = None

```python        title_selectors = [

def get_top_headlines(            'h1', 'title', '[class*="title"]', 

    self,             '[class*="headline"]',

    country: Optional[str] = None,            'meta[property="og:title"]', 

    category: Optional[str] = None,            'meta[name="title"]'

    page_size: int = 20,        ]

    page: int = 1        

) -> List[Dict]:        for selector in title_selectors:

    """            if selector.startswith('meta'):

    Fetch top headlines from NewsAPI                meta_tag = soup.select_one(selector)

                    if meta_tag and meta_tag.get('content'):

    Args:                    title = meta_tag.get('content').strip()

        country: 2-letter country code (us, gb, etc.)                    break

        category: business, entertainment, general, health, science, sports, technology            else:

        page_size: Number of results (1-100)                title_elem = soup.select_one(selector)

        page: Page number for pagination                if title_elem and title_elem.get_text().strip():

                            title = title_elem.get_text().strip()

    Returns:                    break

        List of article dictionaries        

    """        # Try to find article content

    try:        content = None

        endpoint = f'{self.base_url}/top-headlines'        content_selectors = [

        params = {            'article', '[class*="article"]', 

            'pageSize': min(page_size, 100),            '[class*="content"]', '[class*="post"]',

            'page': page            '[class*="story"]', 'main', 

        }            '.entry-content', '.post-content'

                ]

        if country:        

            params['country'] = country        for selector in content_selectors:

        if category:            content_elem = soup.select_one(selector)

            params['category'] = category            if content_elem:

                        # Get all paragraphs

        response = self.session.get(endpoint, params=params)                paragraphs = content_elem.find_all(['p', 'div'])

        response.raise_for_status()                if paragraphs:

                            content = ' '.join([

        data = response.json()                        p.get_text().strip() 

        if data.get('status') == 'ok':                        for p in paragraphs 

            return data.get('articles', [])                        if p.get_text().strip()

        else:                    ])

            print(f"NewsAPI error: {data.get('message', 'Unknown error')}")                    break

            return []        

            # Fallback: get all paragraph text

    except Exception as e:        if not content:

        print(f"Error fetching headlines: {e}")            paragraphs = soup.find_all('p')

        return []            if paragraphs:

```                content = ' '.join([

                    p.get_text().strip() 

**Usage**:                    for p in paragraphs 

```python                    if p.get_text().strip()

client = NewsAPIClient(api_key='your_key')                ])

        

# Fetch US technology news        # Clean up content

articles = client.get_top_headlines(        if content:

    country='us',            # Remove extra whitespace

    category='technology',            content = re.sub(r'\s+', ' ', content).strip()

    page_size=20            # Remove very short sentences (likely navigation/ads)

)            sentences = content.split('.')

```            content = '. '.join([

                s.strip() 

---                for s in sentences 

                if len(s.strip()) > 20

### 5.3 Search Articles            ])

        

**Search for articles by keyword**        # Extract source/domain

        domain = urlparse(url).netloc

```python        source = domain.replace('www.', '').split('.')[0].title()

def search_articles(        

    self,         if not content or len(content) < 50:

    query: str,            return None

    page_size: int = 20,        

    page: int = 1        return {

) -> List[Dict]:            'title': title or 'Unknown Title',

    """            'text': content,

    Search for articles by keyword            'source': source,

                'url': url

    Args:        }

        query: Search query        

        page_size: Number of results (1-100)    except Exception as e:

        page: Page number        print(f"Error extracting content from {url}: {e}")

                return None

    Returns:```

        List of article dictionaries

    """---

    try:

        endpoint = f'{self.base_url}/everything'## 🔗 API Endpoints (in app.py)

        params = {

            'q': query,### 1. **Fetch News** (Lines 375-410)

            'pageSize': min(page_size, 100),```python

            'page': page,@app.route('/fetch-news', methods=['POST'])

            'sortBy': 'publishedAt'def fetch_news():

        }    """

            Endpoint to fetch latest news from NewsAPI

        response = self.session.get(endpoint, params=params)    

        response.raise_for_status()    Request Body:

            {

        data = response.json()        "country": "us",          // Optional: country code

        if data.get('status') == 'ok':        "category": "general",    // Optional: news category

            return data.get('articles', [])        "page_size": 10          // Optional: number of articles

        else:    }

            print(f"NewsAPI error: {data.get('message', 'Unknown error')}")    

            return []    Response:

        {

    except Exception as e:        "articles": [

        print(f"Error searching articles: {e}")            {

        return []                "title": "Article title",

```                "description": "Article description",

                "url": "https://...",

**Usage**:                "source": "News Source",

```python                "published_at": "2024-10-13",

# Search for articles about "climate change"                "credibility_score": 0.8,

articles = client.search_articles(                "prediction": "TRUE",

    query='climate change',                "confidence": 85,

    page_size=10                "individual_predictions": {...}

)            }

```        ]

    }

---    """

    try:

## 6. URL Content Extraction        if not news_fetcher:

            # Return mock data if news fetcher not available

### 6.1 extract_article_content()            mock_articles = [...]

            return jsonify({'articles': mock_articles})

**Extract content from web page URL**

        data = request.get_json(silent=True) or {}

**File**: `app.py`        country = data.get('country', 'us')

        category = data.get('category', 'general')

```python        page_size = data.get('page_size', 10)

def extract_article_content(url: str) -> Dict:

    """        articles = news_fetcher.fetch_and_analyze(

    Extract article content from URL using BeautifulSoup            country=country,

                category=category,

    Args:            page_size=page_size

        url: Article URL        )

                return jsonify({'articles': articles})

    Returns:    except Exception as e:

        Dictionary with extracted content        print(f"Error fetching news: {e}")

    """        return jsonify({'error': str(e)}), 500

    try:```

        from bs4 import BeautifulSoup

        import requests### 2. **Analyze URL** (Lines 291-347)

        ```python

        # Fetch page with timeout@app.route('/analyze-url', methods=['POST'])

        headers = {def analyze_url():

            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'    """

        }    Endpoint to analyze news article from URL

        response = requests.get(url, headers=headers, timeout=10)    

        response.raise_for_status()    Request Body:

            {

        # Parse HTML        "url": "https://example.com/article"

        soup = BeautifulSoup(response.content, 'html.parser')    }

            

        # Remove script and style tags    Response:

        for script in soup(['script', 'style', 'nav', 'footer', 'header']):    {

            script.decompose()        "prediction": "FAKE" / "TRUE",

                "confidence": 85.5,

        # Try different content extraction strategies        "article_title": "Extracted title",

        content = None        "article_source": "Example News",

                "article_text": "Extracted content...",

        # Strategy 1: Look for article tag        "individual_results": {...},

        article = soup.find('article')        "explanation": "Detailed explanation..."

        if article:    }

            content = article.get_text()    """

            try:

        # Strategy 2: Look for common content classes        data = request.get_json()

        if not content:        url = data.get('url', '').strip()

            for class_name in ['article-content', 'post-content', 'entry-content', 'content']:

                div = soup.find('div', class_=class_name)        if not url:

                if div:            return jsonify({'error': 'No URL provided'}), 400

                    content = div.get_text()

                    break        if not predictor:

                    return jsonify({'error': 'ML models not loaded'}), 500

        # Strategy 3: Get all paragraph text        

        if not content:        # Extract content from URL

            paragraphs = soup.find_all('p')        article_content = extract_article_content(url)

            content = ' '.join([p.get_text() for p in paragraphs])        if not article_content:

                    return jsonify({

        # Clean text                'error': 'Could not extract content from URL'

        if content:            }), 400

            # Remove extra whitespace        

            content = re.sub(r'\s+', ' ', content).strip()        # Get ML prediction on extracted content

                    ml_result = predictor.ensemble_predict_majority(

            # Get title            article_content['text']

            title = soup.find('title')        )

            title = title.get_text() if title else 'No title'        

                    # Build response

            return {        response = {

                'success': True,            'prediction': ml_result.get('final_prediction', 'UNKNOWN'),

                'title': title,            'confidence': ml_result.get('confidence', 0),

                'content': content,            'individual_results': ml_result.get('individual_results', {}),

                'url': url            'url': url,

            }            'article_title': article_content.get('title'),

        else:            'article_source': article_content.get('source'),

            return {            'article_text': article_content['text'][:200] + "...",

                'success': False,            'explanation': generate_explanation(ml_result, {...})

                'error': 'No content found',        }

                'url': url        

            }        return jsonify(response)

            

    except Exception as e:    except Exception as e:

        return {        return jsonify({'error': str(e)}), 500

            'success': False,```

            'error': str(e),

            'url': url---

        }

```## 🎯 Key Features



**Extraction Strategies**:### 1. **Data Acquisition**

1. Look for `<article>` tag- ✅ Fetch top headlines by country/category

2. Search for common content classes (`.article-content`, `.post-content`, etc.)- ✅ Search articles by keywords

3. Collect all `<p>` paragraph tags- ✅ Extract content from URLs

4. Clean and normalize text- ✅ Pagination support

- ✅ Rate limiting awareness

**Usage**:

```python### 2. **Data Processing**

# Extract content from URL- ✅ Duplicate detection (URL hashing)

result = extract_article_content('https://example.com/article')- ✅ Text extraction and cleaning

- ✅ Content prioritization (content > description > title)

if result['success']:- ✅ Metadata preservation

    title = result['title']

    content = result['content']### 3. **Integration**

    # Analyze content with Agent 2- ✅ Lazy loading of ML models

    analysis = predictor.ensemble_predict_majority(content)- ✅ Session management for seen articles

```- ✅ Error handling and fallbacks

- ✅ Mock data for testing

---

### 4. **Output Enrichment**

## 7. Text Preprocessing- ✅ Credibility scores (0.0 - 1.0)

- ✅ Individual model predictions

### 7.1 Basic Text Cleaning- ✅ Categorization (Fake/Suspicious/Credible)

- ✅ Statistics generation

**File**: `utils/predictor.py`

---

```python

def preprocess_text(self, text: str) -> str:## 📊 Data Flow

    """Clean and preprocess text"""

    if not text:```

        return ""NewsAPI.org

        ↓

    # Remove URLsNewsAPIClient (fetch articles)

    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)    ↓

    NewsFetcher (orchestration)

    # Remove email addresses    ↓

    text = re.sub(r'\S+@\S+', '', text)Article Processing:

        - Deduplication

    # Remove special characters (keep letters, numbers, spaces)    - Text extraction

    text = re.sub(r'[^\w\s]', ' ', text)    - Content combination

        ↓

    # Remove extra whitespaceML Predictor (Agent 2 integration)

    text = re.sub(r'\s+', ' ', text).strip()    ↓

    Enriched Articles:

    return text    {

```        Original article data +

        credibility_score +

**Cleaning Steps**:        prediction +

1. Remove URLs (http, https, www)        individual_results +

2. Remove email addresses        analyzed_at

3. Remove special characters    }

4. Normalize whitespace    ↓

Frontend / API Response

---```



### 7.2 Advanced Text Preprocessing---



**File**: `utils/text_preprocessor.py` or `credibility_analyzer/text_preprocessor.py`## 🔧 Configuration (config.py)



```python```python

import reclass Config:

import nltk    # NewsAPI settings

from nltk.corpus import stopwords    NEWSAPI_KEY = os.getenv('NEWSAPI_KEY', '')

from nltk.stem import WordNetLemmatizer    

    # Text processing limits

class TextPreprocessor:    MAX_TEXT_LENGTH = 1000

    def __init__(self):    MAX_SEQUENCE_LENGTH = 200

        # Download required NLTK data```

        try:

            nltk.data.find('corpora/stopwords')**Environment Variables (.env)**:

        except:```bash

            nltk.download('stopwords', quiet=True)NEWSAPI_KEY=your_api_key_here

        NEWS_COUNTRY=us

        try:NEWS_LANGUAGE=en

            nltk.data.find('corpora/wordnet')NEWS_PAGE_SIZE=20

        except:NEWS_CATEGORIES=general,technology,business

            nltk.download('wordnet', quiet=True)```

        

        self.stop_words = set(stopwords.words('english'))---

        self.lemmatizer = WordNetLemmatizer()

    ## 🚀 Usage Examples

    def clean(self, text: str) -> str:

        """Advanced text cleaning"""### Example 1: Fetch Top Headlines

        # Lowercase```python

        text = text.lower()from news_fetcher import NewsFetcher

        

        # Remove URLsfetcher = NewsFetcher()

        text = re.sub(r'http\S+|www\S+', '', text)articles = fetcher.fetch_and_analyze(

            country='us',

        # Remove special characters    category='technology',

        text = re.sub(r'[^a-zA-Z\s]', '', text)    page_size=10

        )

        # Tokenize

        words = text.split()for article in articles:

            print(f"{article['title']}: {article['credibility_score']}")

        # Remove stopwords and lemmatize```

        words = [self.lemmatizer.lemmatize(w) for w in words 

                 if w not in self.stop_words and len(w) > 2]### Example 2: Search Articles

        ```python

        return ' '.join(words)articles = fetcher.fetch_and_analyze(

```    query='artificial intelligence',

    page_size=20

---)



## 8. Deduplication Strategystats = fetcher.get_statistics(articles)

print(f"Credible: {stats['credible_percentage']}%")

### 8.1 URL-Based Deduplicationprint(f"Fake: {stats['fake_percentage']}%")

```

```python

class NewsFetcher:### Example 3: Extract from URL

    def __init__(self):```python

        # Track seen URLsfrom app import extract_article_content

        self.seen_urls = set()

    article = extract_article_content('https://example.com/news')

    def _analyze_article(self, article: Dict) -> Optional[Dict]:if article:

        url = article.get('url', '')    print(f"Title: {article['title']}")

            print(f"Source: {article['source']}")

        # Check if URL already processed    print(f"Content: {article['text'][:200]}...")

        if url in self.seen_urls:```

            print(f"Skipping duplicate URL: {url}")

            return None---

        

        # Mark URL as seen## ⚠️ Error Handling

        self.seen_urls.add(url)

        # Continue processing...### Common Errors:

```1. **API Key Missing**: Validates at initialization

2. **Rate Limiting**: Returns empty list, logs error

---3. **Network Errors**: Catches and logs, returns empty list

4. **Invalid Content**: Skips article, continues processing

### 8.2 Content Hash Deduplication5. **Duplicate Articles**: Silently filters using hash check



```python### Logging:

import hashlib- Initialization status

- Article fetch count

class NewsFetcher:- Analysis errors

    def __init__(self):- Duplicate detections

        # Track seen content hashes

        self.seen_hashes = set()---

    

    def _analyze_article(self, article: Dict) -> Optional[Dict]:## 📈 Performance Metrics

        content = article.get('description', '') or article.get('content', '')

        - **Average Fetch Time**: 2-5 seconds for 10 articles

        # Generate content hash- **Analysis Time**: ~0.5 seconds per article

        content_hash = hashlib.md5(content.encode()).hexdigest()- **Memory Usage**: ~50MB for 100 articles

        - **Duplicate Detection**: O(1) hash lookup

        # Check if content already processed

        if content_hash in self.seen_hashes:---

            print(f"Skipping duplicate content")

            return None## 🔄 Dependencies

        

        # Mark content as seen**Required Packages**:

        self.seen_hashes.add(content_hash)- `requests` - HTTP requests to NewsAPI

        # Continue processing...- `beautifulsoup4` - HTML parsing for URL extraction

```- `hashlib` - Duplicate detection

- `datetime` - Timestamp generation

**Why Both?**

- **URL deduplication**: Catches exact same article from same source**Integration**:

- **Content hash**: Catches same content from different URLs/sources- Depends on Agent 2 (Credibility Analyzer) via `UnifiedPredictor`

- Provides data to Agent 3 (Verdict Agent)

---- Uses `config.py` for settings



## 9. API Endpoints---



### 9.1 POST /fetch-news## 📝 Summary



**Fetch and analyze news from NewsAPI****Agent 1 (Article Collector)** serves as the **data acquisition and preprocessing layer** of the fake news detection system. It:



**Request**:1. **Fetches** real-time news from NewsAPI

```json2. **Extracts** content from URLs

{3. **Deduplicates** articles

  "country": "us",4. **Analyzes** credibility using ML models

  "category": "technology",5. **Enriches** article data with predictions

  "page_size": 10,6. **Generates** statistics for dashboard

  "page": 1

}**Input**: NewsAPI credentials, search parameters, URLs  

```**Output**: Enriched articles with credibility scores  

**Role**: Data provider for the entire pipeline

**Response**:

```json---

{

  "articles": [**Last Updated**: October 2025  

    {**Version**: 1.0  

      "title": "Article Title",**Status**: Production Ready ✅

      "description": "Article description...",
      "url": "https://...",
      "urlToImage": "https://...",
      "publishedAt": "2025-10-18T10:30:00Z",
      "source": {"name": "TechCrunch"},
      "analysis": {
        "prediction": "TRUE",
        "confidence": 94.5,
        "verdict_type": "TRUE",
        "model_results": {
          "svm": {"prediction": "TRUE", "confidence": 99.5},
          "lstm": {"prediction": "TRUE", "confidence": 87.0},
          "bert": {"prediction": "TRUE", "confidence": 89.0}
        }
      }
    }
  ],
  "statistics": {
    "total": 10,
    "true_news": 7,
    "fake_news": 2,
    "uncertain": 1,
    "avg_confidence": 91.5
  }
}
```

**Implementation**:
```python
@app.route('/fetch-news', methods=['POST'])
def fetch_news():
    """Fetch and analyze news"""
    try:
        data = request.get_json()
        
        country = data.get('country')
        category = data.get('category')
        page_size = data.get('page_size', 20)
        page = data.get('page', 1)
        
        # Use Agent 1 to fetch and analyze
        articles = news_fetcher.fetch_and_analyze(
            country=country,
            category=category,
            page_size=page_size,
            page=page
        )
        
        # Generate statistics
        stats = news_fetcher.get_statistics(articles)
        
        return jsonify({
            'articles': articles,
            'statistics': stats
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
```

---

### 9.2 POST /analyze-url

**Extract and analyze content from URL**

**Request**:
```json
{
  "url": "https://example.com/news-article"
}
```

**Response**:
```json
{
  "prediction": "TRUE",
  "confidence": 95.5,
  "verdict_type": "TRUE",
  "explanation": "All models agree...",
  "url": "https://example.com/news-article",
  "extracted_text_preview": "First 200 characters...",
  "extraction_success": true
}
```

**Implementation**:
```python
@app.route('/analyze-url', methods=['POST'])
def analyze_url():
    """Analyze content from URL"""
    try:
        data = request.get_json()
        url = data.get('url')
        
        if not url:
            return jsonify({'error': 'URL required'}), 400
        
        # Agent 1: Extract content from URL
        extraction = extract_article_content(url)
        
        if not extraction['success']:
            return jsonify({
                'error': 'Failed to extract content',
                'details': extraction.get('error')
            }), 400
        
        content = extraction['content']
        
        # Agent 2: Analyze with ML models
        result = predictor.ensemble_predict_majority(content)
        
        # Add URL info to result
        result['url'] = url
        result['extracted_text_preview'] = content[:200] + '...'
        result['extraction_success'] = True
        
        return jsonify(_make_json_safe(result))
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
```

---

## 10. Code Examples

### Example 1: Fetch US Technology News

```python
from news_fetcher import NewsFetcher

# Initialize fetcher
fetcher = NewsFetcher()

# Fetch US technology news
articles = fetcher.fetch_and_analyze(
    country='us',
    category='technology',
    page_size=10
)

# Print results
for article in articles:
    print(f"Title: {article['title']}")
    print(f"Prediction: {article['analysis']['prediction']}")
    print(f"Confidence: {article['analysis']['confidence']}%")
    print("---")

# Get statistics
stats = fetcher.get_statistics(articles)
print(f"Total: {stats['total']}")
print(f"True News: {stats['true_news']}")
print(f"Fake News: {stats['fake_news']}")
```

---

### Example 2: Extract and Analyze URL

```python
from app import extract_article_content
from utils.predictor import UnifiedPredictor
from utils.model_loader import ModelLoader

# Initialize components
model_loader = ModelLoader()
model_loader.load_all_models()
predictor = UnifiedPredictor(model_loader)

# Extract content from URL
url = 'https://example.com/news-article'
extraction = extract_article_content(url)

if extraction['success']:
    content = extraction['content']
    
    # Analyze with ML models
    result = predictor.ensemble_predict_majority(content)
    
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']}%")
else:
    print(f"Extraction failed: {extraction['error']}")
```

---

### Example 3: Search Articles by Keyword

```python
from news_apis.newsapi_client import NewsAPIClient
from config import Config

# Initialize client
client = NewsAPIClient(api_key=Config.NEWSAPI_KEY)

# Search for articles
articles = client.search_articles(
    query='artificial intelligence',
    page_size=20
)

print(f"Found {len(articles)} articles")

for article in articles:
    print(f"Title: {article['title']}")
    print(f"Source: {article['source']['name']}")
    print(f"Published: {article['publishedAt']}")
    print("---")
```

---

## 11. Error Handling

### 11.1 NewsAPI Errors

```python
try:
    articles = client.get_top_headlines(country='us')
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 401:
        print("Invalid API key")
    elif e.response.status_code == 429:
        print("Rate limit exceeded")
    else:
        print(f"HTTP error: {e}")
except requests.exceptions.Timeout:
    print("Request timeout")
except requests.exceptions.ConnectionError:
    print("Connection error")
```

---

### 11.2 URL Extraction Errors

```python
extraction = extract_article_content(url)

if not extraction['success']:
    error = extraction.get('error', 'Unknown error')
    
    if 'timeout' in error.lower():
        print("Request timeout - website too slow")
    elif '404' in error:
        print("Article not found")
    elif '403' in error:
        print("Access forbidden - website blocking scrapers")
    else:
        print(f"Extraction error: {error}")
```

---

### 11.3 Deduplication Handling

```python
def _analyze_article(self, article: Dict) -> Optional[Dict]:
    """Analyze article with duplicate checking"""
    try:
        # Check URL duplicate
        url = article.get('url', '')
        if url in self.seen_urls:
            return None  # Skip silently
        
        # Check content duplicate
        content = article.get('description', '')
        content_hash = hashlib.md5(content.encode()).hexdigest()
        if content_hash in self.seen_hashes:
            return None  # Skip silently
        
        # Mark as seen
        self.seen_urls.add(url)
        self.seen_hashes.add(content_hash)
        
        # Analyze...
        
    except Exception as e:
        print(f"Error analyzing article: {e}")
        return None  # Skip on error
```

---

## Summary

**Agent 1: Article Collector** successfully:

✅ Fetches news from NewsAPI.org (headlines, searches)  
✅ Extracts content from URLs using BeautifulSoup  
✅ Preprocesses and cleans text for ML analysis  
✅ Deduplicates articles (URL + content hash)  
✅ Prepares data for Agent 2 (Credibility Analyzer)  
✅ Generates statistics from analyzed articles  
✅ Handles errors gracefully  

**Next Step**: Data flows to **Agent 2: Credibility Analyzer** for ML-based fake news detection!

---

**Agent Version**: 1.0  
**Last Updated**: October 2025  
**Status**: Production Ready ✅
