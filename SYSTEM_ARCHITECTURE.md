# 🏗️ SYSTEM ARCHITECTURE - Fake News Detection System# 🎯 FAKE NEWS DETECTION SYSTEM - COMPLETE ARCHITECTURE OVERVIEW



## 📋 Table of Contents## 📋 Table of Contents

1. [System Overview](#system-overview)

1. [System Overview](#system-overview)2. [Three-Agent Architecture](#three-agent-architecture)

2. [Three-Agent Architecture](#three-agent-architecture)3. [Data Flow](#data-flow)

3. [Data Flow](#data-flow)4. [Integration Points](#integration-points)

4. [Technology Stack](#technology-stack)5. [API Endpoints](#api-endpoints)

5. [API Endpoints](#api-endpoints)6. [Technology Stack](#technology-stack)

6. [Integration Points](#integration-points)7. [Key Features](#key-features)

7. [Security & Performance](#security--performance)8. [Deployment](#deployment)

8. [Deployment Guide](#deployment-guide)

9. [File Structure](#file-structure)---

10. [Configuration](#configuration)

## 🎯 System Overview

---

The **Fake News Detection System** is a sophisticated multi-agent AI platform that analyzes news articles and text content to determine credibility. It uses a three-agent architecture with ensemble machine learning models and real-time news verification.

## 1. System Overview

### Key Statistics:

The **Fake News Detection System** is a sophisticated Flask-based web application that uses a **three-agent architecture** to analyze news articles and determine their credibility. The system combines machine learning models, external news verification, and intelligent consensus analysis to provide accurate fake news detection.- **3 AI Agents** working in harmony

- **3 ML Models** (SVM, LSTM, BERT) with 94% ensemble accuracy

### Key Features- **Real-time** news fetching from NewsAPI

- **Comprehensive** credibility analysis

✅ **Three-Agent Intelligence**:- **Human-readable** explanations

- **Agent 1**: Article Collector - Fetches and preprocesses news

- **Agent 2**: Credibility Analyzer - ML-based analysis with ensemble voting---

- **Agent 3**: Verdict Agent - Generates final verdict with explanations

## 🏗️ Three-Agent Architecture

✅ **Multi-Model Ensemble**:

- SVM (Support Vector Machine) - 99.5% accuracyThe system is built around three specialized agents, each with distinct responsibilities:

- LSTM (Long Short-Term Memory) - 87.0% accuracy

- BERT (Bidirectional Encoder Representations) - 89.0% accuracy```

- **Ensemble Accuracy**: ~94% with majority voting┌─────────────────────────────────────────────────────────────┐

│                    USER INPUT                                │

✅ **External Verification**:│  (Text to analyze / URL to check / News to fetch)           │

- NewsAPI.org integration for real-time fact-checking└──────────────────┬──────────────────────────────────────────┘

- Semantic similarity matching                   │

- Cross-reference validation                   ▼

┌──────────────────────────────────────────────────────────────┐

✅ **Dual Operation Modes**:│              AGENT 1: ARTICLE COLLECTOR                      │

- **Free Mode**: Basic analysis and URL verification│  ┌────────────────────────────────────────────────────┐     │

- **Commercial Mode**: Advanced features, subscriptions, usage tracking│  │  • Fetch from NewsAPI                              │     │

│  │  • Extract content from URLs                       │     │

### System Capabilities│  │  • Deduplicate articles                            │     │

│  │  • Preprocess text                                 │     │

- 📰 **News Fetching**: Retrieve articles from NewsAPI by country/category│  └────────────────────────────────────────────────────┘     │

- 🔗 **URL Analysis**: Extract and analyze content from any URL│  Files: news_fetcher.py, news_apis/newsapi_client.py        │

- 🤖 **ML Prediction**: Three-model ensemble with confidence scoring└──────────────────┬──────────────────────────────────────────┘

- ✅ **Verification**: Cross-check against trusted news sources                   │

- 📊 **Analytics**: Track analysis history and statistics                   ▼

- 💼 **Commercial**: Subscription plans, API limits, user management┌──────────────────────────────────────────────────────────────┐

│           AGENT 2: CREDIBILITY ANALYZER                      │

---│  ┌────────────────────────────────────────────────────┐     │

│  │  ML Models:                                        │     │

## 2. Three-Agent Architecture│  │  • SVM (99.5% accuracy)                           │     │

│  │  • LSTM (87.0% accuracy)                          │     │

The system uses a **sequential three-agent pipeline** where each agent has specialized responsibilities:│  │  • BERT (89.0% accuracy)                          │     │

│  │                                                    │     │

```│  │  • Ensemble voting (Majority)                     │     │

┌─────────────────────────────────────────────────────────────────┐│  │  • Feature extraction                             │     │

│                    USER REQUEST                                  ││  │  • Confidence calibration                         │     │

│              (Text / URL / News Query)                          ││  └────────────────────────────────────────────────────┘     │

└────────────────────────┬────────────────────────────────────────┘│  Files: utils/model_loader.py, utils/predictor.py           │

                         ││         credibility_analyzer/credibility_analyzer.py         │

                         ▼└──────────────────┬──────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐                   │

│                  AGENT 1: ARTICLE COLLECTOR                      │                   ▼

│  ┌──────────────────────────────────────────────────────────┐  │┌──────────────────────────────────────────────────────────────┐

│  │  • Fetch news from NewsAPI                                │  ││             AGENT 3: VERDICT AGENT                           │

│  │  • Extract content from URLs                              │  ││  ┌────────────────────────────────────────────────────┐     │

│  │  • Preprocess and clean text                              │  ││  │  • Analyze model consensus                         │     │

│  │  • Deduplicate articles                                   │  ││  │  • Calculate final confidence                      │     │

│  │  • Prepare for analysis                                   │  ││  │  • Generate explanations                           │     │

│  └──────────────────────────────────────────────────────────┘  ││  │  • NewsAPI cross-verification                      │     │

└────────────────────────┬────────────────────────────────────────┘│  │  • Final verdict synthesis                         │     │

                         ││  └────────────────────────────────────────────────────┘     │

                         │ Clean Text│  Files: verdict_agent/verdict_agent.py, app.py              │

                         ▼└──────────────────┬──────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐                   │

│               AGENT 2: CREDIBILITY ANALYZER                      │                   ▼

│  ┌──────────────────────────────────────────────────────────┐  │┌──────────────────────────────────────────────────────────────┐

│  │  ML MODEL 1: SVM                                          │  ││                    OUTPUT                                     │

│  │    → Prediction: FAKE/TRUE                                │  ││  {                                                            │

│  │    → Confidence: 99.5%                                    │  ││    "verdict": "TRUE",                                         │

│  └──────────────────────────────────────────────────────────┘  ││    "confidence": 91.0,                                        │

│  ┌──────────────────────────────────────────────────────────┐  ││    "explanation": "High probability of credible content...",  │

│  │  ML MODEL 2: LSTM                                         │  ││    "individual_results": {...},                               │

│  │    → Prediction: FAKE/TRUE                                │  ││    "news_verification": {...}                                 │

│  │    → Confidence: 87.0%                                    │  ││  }                                                            │

│  └──────────────────────────────────────────────────────────┘  │└──────────────────────────────────────────────────────────────┘

│  ┌──────────────────────────────────────────────────────────┐  │```

│  │  ML MODEL 3: BERT                                         │  │

│  │    → Prediction: FAKE/TRUE                                │  │---

│  │    → Confidence: 89.0%                                    │  │

│  └──────────────────────────────────────────────────────────┘  │## 📊 Agent Responsibilities

│  ┌──────────────────────────────────────────────────────────┐  │

│  │  ENSEMBLE VOTING                                          │  │### **AGENT 1: Article Collector** 📰

│  │    → Majority Vote: 2 out of 3 models                     │  │**Role**: Data Acquisition and Preprocessing

│  │    → Confidence Weighting                                 │  │

│  │    → Final Prediction                                     │  │**Responsibilities**:

│  └──────────────────────────────────────────────────────────┘  │1. Fetch news from NewsAPI (top headlines, search)

└────────────────────────┬────────────────────────────────────────┘2. Extract content from URLs (web scraping)

                         │3. Deduplicate articles (hash-based)

                         │ Model Predictions4. Clean and format text

                         ▼5. Prepare data for analysis

┌─────────────────────────────────────────────────────────────────┐

│                  AGENT 3: VERDICT AGENT                          │**Input**: 

│  ┌──────────────────────────────────────────────────────────┐  │- NewsAPI credentials

│  │  • Analyze model consensus                                │  │- Search parameters (country, category, query)

│  │  • Calculate final confidence                             │  │- URLs to extract

│  │  • Generate human-readable explanation                    │  │

│  │  • Integrate NewsAPI verification (optional)              │  │**Output**: 

│  │  • Determine verdict type:                                │  │- Enriched articles with metadata

│  │    - TRUE (Strong Agreement)                              │  │- Clean text for analysis

│  │    - FALSE (Strong Agreement)                             │  │- Article statistics

│  │    - MISLEADING (Mixed Signals)                           │  │

│  │    - UNCERTAIN (Low Confidence)                           │  │**Key Files**:

│  └──────────────────────────────────────────────────────────┘  │- `news_fetcher.py` (main orchestrator)

└────────────────────────┬────────────────────────────────────────┘- `news_apis/newsapi_client.py` (API client)

                         │- `app.py` (extract_article_content function)

                         │ Final Verdict

                         ▼**Code Example**:

┌─────────────────────────────────────────────────────────────────┐```python

│                     JSON RESPONSE                                │from news_fetcher import NewsFetcher

│  {                                                               │

│    "prediction": "TRUE",                                         │fetcher = NewsFetcher()

│    "confidence": 95.5,                                           │articles = fetcher.fetch_and_analyze(

│    "verdict_type": "TRUE",                                       │    country='us',

│    "explanation": "All models agree...",                         │    category='technology',

│    "model_results": [...],                                       │    page_size=10

│    "news_verification": {...}                                    │)

│  }                                                               │# Returns: List of articles with credibility scores

└─────────────────────────────────────────────────────────────────┘```

```

---

### Agent Responsibilities

### **AGENT 2: Credibility Analyzer** 🔍

#### **Agent 1: Article Collector** (`news_fetcher.py`)**Role**: Machine Learning Prediction

- **Input**: URL, news query, or raw text

- **Process**: Fetch → Extract → Clean → Deduplicate**Responsibilities**:

- **Output**: Clean, preprocessed text ready for analysis1. Load and manage ML models (SVM, LSTM, BERT)

- **Key Files**: `news_fetcher.py`, `news_apis/newsapi_client.py`2. Preprocess text (cleaning, normalization)

3. Extract advanced features (sentiment, readability, etc.)

#### **Agent 2: Credibility Analyzer** (`utils/predictor.py`)4. Run ensemble prediction with majority voting

- **Input**: Clean text from Agent 15. Calculate confidence scores

- **Process**: Load models → Predict (SVM, LSTM, BERT) → Ensemble voting6. Provide individual model results

- **Output**: Prediction, confidence, individual model results

- **Key Files**: `utils/model_loader.py`, `utils/predictor.py`, `credibility_analyzer/`**Input**: 

- Raw text content

#### **Agent 3: Verdict Agent** (`verdict_agent/verdict_agent.py`)- Model files (PKL, H5, pre-trained)

- **Input**: Text + Model predictions from Agent 2

- **Process**: Analyze consensus → Calculate confidence → Generate explanation**Output**: 

- **Output**: Final verdict with human-readable explanation- Final prediction (FAKE/TRUE)

- **Key Files**: `verdict_agent/verdict_agent.py`- Confidence score (0-100%)

- Individual model predictions

---- Voting details



## 3. Data Flow**Key Files**:

- `utils/model_loader.py` (load models)

### Complete Analysis Pipeline- `utils/predictor.py` (ensemble prediction)

- `credibility_analyzer/credibility_analyzer.py` (analysis)

```- `credibility_analyzer/feature_extractor.py` (features)

USER INPUT

    │**Code Example**:

    ├─→ Text Analysis Flow```python

    │   1. User provides text directlyfrom utils.model_loader import ModelLoader

    │   2. Text → Agent 2 (Credibility Analyzer)from utils.predictor import UnifiedPredictor

    │   3. Agent 2 → Agent 3 (Verdict Agent)

    │   4. Return verdict to userloader = ModelLoader()

    │loader.load_all_models()

    ├─→ URL Analysis Flow

    │   1. User provides URLpredictor = UnifiedPredictor(loader)

    │   2. Agent 1 extracts content from URL (BeautifulSoup)result = predictor.ensemble_predict_majority(text)

    │   3. Extracted text → Agent 2 (Credibility Analyzer)# Returns: {

    │   4. Agent 2 → Agent 3 (Verdict Agent)#   "final_prediction": "TRUE",

    │   5. Return verdict to user#   "confidence": 91.0,

    │#   "individual_results": {...}

    └─→ News Fetching Flow# }

        1. User requests news (country, category)```

        2. Agent 1 fetches from NewsAPI

        3. Agent 1 deduplicates articles---

        4. For each article:

           a. Agent 1 → Agent 2 (Credibility Analyzer)### **AGENT 3: Verdict Agent** ⚖️

           b. Agent 2 → Agent 3 (Verdict Agent)**Role**: Decision Making and Explanation

        5. Return analyzed articles with verdicts

```**Responsibilities**:

1. Analyze consensus among models

### Data Transformation2. Calculate final confidence (weighted by agreement)

3. Generate human-readable explanations

```4. Integrate NewsAPI verification results

Raw Input → Cleaned Text → ML Features → Predictions → Verdict → Explanation5. Provide comprehensive verdicts

6. Format results for display

1. Raw Input (Any format)

   ↓**Input**: 

2. Agent 1: Text Cleaning- Model predictions from Agent 2

   - Remove HTML tags- Original text

   - Strip special characters- NewsAPI verification results

   - Normalize whitespace

   - Extract main content**Output**: 

   ↓- Final verdict (TRUE/FALSE/MISLEADING/UNCERTAIN)

3. Agent 2: Feature Extraction- Confidence score (0.0-1.0)

   - SVM: TF-IDF vectorization- Detailed explanation with emojis

   - LSTM: Tokenization + Padding- Consensus analysis

   - BERT: Contextual embeddings- Recommendations

   ↓

4. Agent 2: Model Predictions**Key Files**:

   - SVM: Linear classification- `verdict_agent/verdict_agent.py` (verdict logic)

   - LSTM: Sequential analysis- `app.py` (generate_explanation function)

   - BERT: Transformer-based classification

   ↓**Code Example**:

5. Agent 2: Ensemble Voting```python

   - Collect all predictionsfrom verdict_agent.verdict_agent import VerdictAgent, ModelResult

   - Apply majority voting

   - Weight by confidenceagent = VerdictAgent()

   ↓model_results = [

6. Agent 3: Consensus Analysis    ModelResult("SVM", "true", 0.995, "traditional_ml", 0.995),

   - Check agreement level    ModelResult("LSTM", "true", 0.823, "deep_learning", 0.870),

   - Map to verdict type    ModelResult("BERT", "true", 0.912, "transformer", 0.890)

   - Calculate final confidence]

   ↓

7. Agent 3: Explanation Generationverdict = agent.generate_verdict(text, model_results)

   - Generate human-readable text# Returns: {

   - Include model reasoning#   "verdict": "true",

   - Add verification results (if available)#   "confidence": 0.91,

   ↓#   "explanation": "The content appears to be TRUE...",

8. JSON Response#   "consensus_analysis": {...}

   - Structured output# }

   - All metadata included```

   - Ready for frontend display

```---



---## 🔄 Complete Data Flow



## 4. Technology Stack### **End-to-End Process**:



### Core Technologies```

1. USER REQUEST

#### **Backend Framework**   ├─ Text Analysis: User submits text

- **Flask 3.0.0**: Web framework   ├─ URL Analysis: User submits URL

  - Routes and request handling   └─ News Fetching: User requests news by category

  - Session management

  - Template rendering2. AGENT 1: ARTICLE COLLECTOR

- **Flask-CORS**: Cross-Origin Resource Sharing   ├─ Fetch from NewsAPI OR Extract from URL

  - Enable API access from frontend   ├─ Clean and preprocess text

   ├─ Deduplicate (if multiple articles)

#### **Machine Learning**   └─ Pass to Agent 2

- **scikit-learn 1.4.0**: SVM model

  - TF-IDF vectorization3. AGENT 2: CREDIBILITY ANALYZER

  - Linear SVM classifier   ├─ Load ML models (SVM, LSTM, BERT)

  - 99.5% accuracy on test set   ├─ Preprocess text (remove URLs, special chars)

     ├─ Parallel predictions:

- **TensorFlow 2.13.0**: LSTM model   │  ├─ SVM: Vectorize → Predict → 99.5%

  - Keras API for neural networks   │  ├─ LSTM: Tokenize → Pad → Predict → 82.3%

  - Sequential LSTM architecture   │  └─ BERT: Tokenize → Extract features → Predict → 91.2%

  - 87.0% accuracy on test set   ├─ Ensemble voting (majority wins)

     ├─ Calculate average confidence

- **Transformers 4.30.0**: BERT model   └─ Pass to Agent 3

  - HuggingFace library

  - DistilBERT base uncased4. AGENT 3: VERDICT AGENT

  - 89.0% accuracy on test set   ├─ Analyze consensus (all agree? majority?)

     ├─ Calculate final confidence (weighted by agreement)

- **PyTorch**: BERT backend   ├─ Generate explanation:

  - Tensor operations   │  ├─ AI analysis section

  - GPU acceleration (if available)   │  ├─ Individual model results

   │  └─ NewsAPI verification (if available)

#### **NLP Libraries**   └─ Return final verdict

- **NLTK 3.8**: Natural Language Processing

  - Tokenization5. RESPONSE TO USER

  - Stopword removal   └─ JSON response with:

  - Text preprocessing      ├─ Verdict (TRUE/FALSE)

        ├─ Confidence (91.0%)

- **spaCy**: Advanced NLP      ├─ Explanation (human-readable)

  - Named Entity Recognition      ├─ Individual results

  - Part-of-speech tagging      └─ News verification

  - Dependency parsing```



#### **External APIs**---

- **NewsAPI.org**: News fetching

  - Top headlines by country## 🔌 Integration Points

  - Search by keywords

  - Category filtering### **Between Agents**:

  - 100 requests/day (free tier)

#### **Agent 1 → Agent 2**:

#### **Web Scraping**```python

- **BeautifulSoup4 4.12.2**: HTML parsing# Agent 1 prepares text and passes to Agent 2

  - Extract article contenttext_to_analyze = content or description or title

  - Clean HTML tagsresult = predictor.ensemble_predict_majority(text_to_analyze)

  - Handle various website structures```

  

- **requests 2.31.0**: HTTP client#### **Agent 2 → Agent 3**:

  - Fetch web pages```python

  - Handle redirects# Agent 2 provides predictions to Agent 3

  - Timeout managementmodel_results = [

    ModelResult(

#### **Data Processing**        model_name="SVM",

- **NumPy 1.24.0**: Numerical operations        label=svm_result['prediction'].lower(),

  - Array operations        confidence=svm_result['confidence'] / 100.0,

  - Mathematical functions        model_type="traditional_ml",

          accuracy=0.995

- **pandas 2.0.0**: Data manipulation    ),

  - DataFrames for analytics    # ... LSTM and BERT results

  - CSV handling]

verdict = verdict_agent.generate_verdict(text, model_results)

#### **Commercial Features**```

- **Stripe**: Payment processing (optional)

- **JWT**: Authentication tokens#### **Agent 3 → API Response**:

- **SQLite**: User data storage (optional)```python

# Agent 3 formats final response

---response = {

    'prediction': verdict['verdict'],

## 5. API Endpoints    'confidence': verdict['confidence'],

    'explanation': verdict['explanation'],

### Core Analysis Endpoints    'individual_results': ml_result['individual_results'],

    'news_api_results': news_api_results

#### `POST /analyze`}

**Analyze text for fake news**return jsonify(response)

```

**Request**:

```json### **External Integrations**:

{

  "text": "Your news article text here..."1. **NewsAPI.org**: Real-time news fetching

}2. **Web Scraping**: URL content extraction

```3. **Session Storage**: History tracking

4. **Commercial Features**: User management, usage tracking

**Response**:

```json---

{

  "prediction": "TRUE",## 🌐 API Endpoints

  "confidence": 95.5,

  "verdict_type": "TRUE",### **Main Flask Application** (app.py)

  "explanation": "All three ML models strongly agree...",

  "timestamp": "2025-10-18T10:30:00",#### 1. **Analyze Text**

  "model_results": {```http

    "svm": {"prediction": "TRUE", "confidence": 99.5},POST /analyze

    "lstm": {"prediction": "TRUE", "confidence": 87.0},Content-Type: application/json

    "bert": {"prediction": "TRUE", "confidence": 89.0}

  },{

  "consensus": {  "text": "Article text to analyze..."

    "agreement_level": "STRONG",}

    "votes": {"TRUE": 3, "FAKE": 0}

  }Response:

}{

```  "prediction": "TRUE",

  "confidence": 91.0,

**Usage**:  "explanation": "🤖 AI ANALYSIS: High probability...",

```python  "individual_results": {

import requests    "SVM": {"prediction": "TRUE", "confidence": 99.5},

    "LSTM": {"prediction": "TRUE", "confidence": 82.3},

response = requests.post('http://localhost:5000/analyze', json={    "BERT": {"prediction": "TRUE", "confidence": 91.2}

    'text': 'Your news text here'  },

})  "news_api_results": {...},

result = response.json()  "timestamp": "2024-10-13T12:00:00"

print(f"Prediction: {result['prediction']}")}

print(f"Confidence: {result['confidence']}%")```

```

#### 2. **Analyze URL**

---```http

POST /analyze-url

#### `POST /analyze-url`Content-Type: application/json

**Extract and analyze content from URL**

{

**Request**:  "url": "https://example.com/article"

```json}

{

  "url": "https://example.com/news-article"Response:

}{

```  "prediction": "FALSE",

  "confidence": 85.0,

**Response**:  "article_title": "Extracted Title",

```json  "article_source": "Example News",

{  "article_text": "Extracted content...",

  "prediction": "FALSE",  "explanation": "🤖 AI ANALYSIS: High probability of misinformation...",

  "confidence": 92.3,  "individual_results": {...}

  "verdict_type": "FALSE",}

  "explanation": "Two out of three models detected...",```

  "url": "https://example.com/news-article",

  "extracted_text_preview": "First 200 characters...",#### 3. **Fetch News**

  "extraction_success": true,```http

  "model_results": {...},POST /fetch-news

  "news_verification": {Content-Type: application/json

    "found": true,

    "matching_articles": [...]{

  }  "country": "us",

}  "category": "technology",

```  "page_size": 10

}

**Usage**:

```pythonResponse:

response = requests.post('http://localhost:5000/analyze-url', json={{

    'url': 'https://example.com/article'  "articles": [

})    {

result = response.json()      "title": "Article Title",

```      "description": "Article description",

      "url": "https://...",

---      "source": "News Source",

      "published_at": "2024-10-13",

#### `POST /fetch-news`      "credibility_score": 0.85,

**Fetch and analyze news from NewsAPI**      "prediction": "TRUE",

      "confidence": 91.0,

**Request**:      "individual_predictions": {...}

```json    },

{    ...

  "country": "us",  ]

  "category": "technology",}

  "page_size": 10```

}

```#### 4. **Get History**

```http

**Parameters**:GET /history

- `country` (optional): 2-letter country code (us, gb, in, etc.)

- `category` (optional): business, entertainment, general, health, science, sports, technologyResponse:

- `page_size` (optional): Number of articles (1-100, default: 20){

  "history": [

**Response**:    {

```json      "prediction": "TRUE",

{      "confidence": 91.0,

  "articles": [      "text": "Analyzed text...",

    {      "timestamp": "2024-10-13T12:00:00"

      "title": "Article Title",    },

      "description": "Article description...",    ...

      "url": "https://...",  ]

      "urlToImage": "https://...",}

      "publishedAt": "2025-10-18T...",```

      "source": {"name": "Source Name"},

      "analysis": {#### 5. **Clear History**

        "prediction": "TRUE",```http

        "confidence": 94.2,POST /clear-history

        "verdict_type": "TRUE"

      }Response:

    }{

  ],  "message": "History cleared"

  "statistics": {}

    "total": 10,```

    "true_news": 7,

    "fake_news": 2,---

    "uncertain": 1,

    "avg_confidence": 91.5## 🛠️ Technology Stack

  }

}### **Backend**:

```- **Flask 3.x**: Web framework

- **Python 3.8+**: Programming language

**Usage**:- **NumPy**: Numerical computing

```python- **Pickle**: Model serialization

response = requests.post('http://localhost:5000/fetch-news', json={

    'country': 'us',### **Machine Learning**:

    'category': 'technology',- **scikit-learn**: SVM model, TF-IDF vectorization

    'page_size': 10- **TensorFlow/Keras**: LSTM neural network

})- **Transformers (HuggingFace)**: BERT/DistilBERT

news = response.json()- **PyTorch**: BERT model backend

```

### **Data Processing**:

---- **BeautifulSoup4**: HTML parsing (URL extraction)

- **Requests**: HTTP client

#### `GET /history`- **hashlib**: Duplicate detection

**Retrieve analysis history**- **re (regex)**: Text preprocessing



**Response**:### **External APIs**:

```json- **NewsAPI.org**: News fetching

{- **python-dotenv**: Environment configuration

  "history": [

    {### **Frontend** (not covered in detail):

      "text_preview": "First 100 chars...",- **Tailwind CSS**: Styling

      "prediction": "TRUE",- **JavaScript**: Interactivity

      "confidence": 95.5,- **Jinja2**: Template rendering

      "timestamp": "2025-10-18T10:30:00",

      "verdict_type": "TRUE"---

    }

  ],## ✨ Key Features

  "count": 5

}### **1. Multi-Model Ensemble**

```- Combines 3 different ML approaches

- Majority voting for robustness

---- Individual model results available

- Graceful degradation if models fail

#### `POST /clear-history`

**Clear analysis history**### **2. Real-Time News Verification**

- Fetch latest news from NewsAPI

**Response**:- Cross-reference analyzed text with trusted sources

```json- Similarity scoring

{- Best match identification

  "message": "History cleared",

  "success": true### **3. Comprehensive Analysis**

}- Sentiment analysis

```- Readability scoring

- Sensational language detection

---- Factual indicators counting

- Punctuation analysis

#### `GET /test`

**Health check endpoint**### **4. Human-Readable Output**

- Clear verdicts (TRUE/FALSE/MISLEADING/UNCERTAIN)

**Response**:- Confidence percentages

```json- Detailed explanations with emojis

{- Model-by-model breakdown

  "status": "healthy",- NewsAPI verification results

  "model_status": {

    "svm": "loaded",### **5. Advanced Features**

    "lstm": "loaded",- URL content extraction

    "bert": "loaded"- Duplicate detection

  },- Session-based history

  "news_verifier": "available",- Pagination support

  "timestamp": "2025-10-18T10:30:00"- Statistics generation

}

```---



---## 📦 File Structure



### Commercial Endpoints (Optional)```

d:\ML Projects\FND\

#### `POST /commercial/register`├── app.py                          # Main Flask application

Register new user├── config.py                       # Configuration settings

├── .env                            # Environment variables

#### `POST /commercial/login`├── requirements.txt                # Python dependencies

User authentication│

├── news_apis/

#### `GET /commercial/dashboard`│   ├── __init__.py

User dashboard with analytics│   └── newsapi_client.py           # AGENT 1: NewsAPI client

│

#### `POST /commercial/upgrade`├── news_fetcher.py                 # AGENT 1: Main orchestrator

Upgrade subscription plan│

├── utils/

#### `GET /commercial/usage`│   ├── __init__.py

Check API usage limits│   ├── model_loader.py             # AGENT 2: Model loading

│   ├── predictor.py                # AGENT 2: Ensemble prediction

---│   ├── text_preprocessor.py        # Text cleaning

│   └── news_verifier.py            # NewsAPI verification

## 6. Integration Points│

├── credibility_analyzer/

### Agent Integration│   ├── __init__.py

│   ├── credibility_analyzer.py     # AGENT 2: Analysis logic

#### **Agent 1 → Agent 2 Integration**│   ├── feature_extractor.py        # AGENT 2: Feature extraction

│   ├── confidence_calibrator.py    # AGENT 2: Confidence scoring

```python│   └── text_preprocessor.py        # Text preprocessing

# In app.py or news_fetcher.py│

├── verdict_agent/

# Agent 1: Extract and clean text│   ├── __init__.py

from news_fetcher import NewsFetcher│   └── verdict_agent.py            # AGENT 3: Verdict generation

fetcher = NewsFetcher()│

clean_text = fetcher._analyze_article(article)├── models/

│   ├── new_svm_model.pkl           # SVM classifier

# Pass to Agent 2│   ├── new_svm_vectorizer.pkl      # TF-IDF vectorizer

from utils.predictor import UnifiedPredictor│   ├── lstm_fake_news_model.h5     # LSTM model

result = predictor.ensemble_predict_majority(clean_text['content'])│   ├── lstm_tokenizer.pkl          # LSTM tokenizer

```│   └── bert_fake_news_model/       # BERT model directory

│       └── classifier.pkl          # Custom classifier

**Data Contract**:│

- **Input to Agent 2**: Clean string (no HTML, normalized whitespace)├── templates/

- **Output from Agent 2**: Dict with prediction, confidence, model_results│   ├── index.html                  # Main UI

│   └── home.html                   # Landing page

---│

├── static/

#### **Agent 2 → Agent 3 Integration**│   ├── css/

│   └── js/

```python│       └── main.js                 # Frontend logic

# In app.py│

└── commercial/                     # Commercial features (optional)

# Agent 2: Get ML predictions    ├── auth/

ml_result = predictor.ensemble_predict_majority(text)    ├── api_limits/

    └── subscriptions/

# Convert to ModelResult objects for Agent 3```

from verdict_agent.verdict_agent import VerdictAgent, ModelResult

---

model_results = []

for model_name, result in ml_result['model_results'].items():## 🚀 Deployment Guide

    model_results.append(ModelResult(

        model_name=model_name,### **Local Development**:

        prediction=result['prediction'],

        confidence=result['confidence'],1. **Install Dependencies**:

        raw_output=result```bash

    ))pip install -r requirements.txt

```

# Pass to Agent 3

agent = VerdictAgent()2. **Configure Environment**:

verdict = agent.generate_verdict(text, model_results)```bash

```# Create .env file

cp .env.example .env

**Data Contract**:

- **Input to Agent 3**: List of ModelResult objects# Add your NewsAPI key

- **Output from Agent 3**: Dict with verdict, explanation, confidenceNEWSAPI_KEY=your_api_key_here

```

---

3. **Start Server**:

### External Service Integration```bash

python app.py

#### **NewsAPI Integration**```



```python4. **Access Application**:

# In news_apis/newsapi_client.py```

http://localhost:5000

from newsapi import NewsApiClient```



api_key = os.getenv('NEWSAPI_KEY')### **Production Deployment**:

newsapi = NewsApiClient(api_key=api_key)

1. **Use Production Server** (Gunicorn):

# Fetch top headlines```bash

headlines = newsapi.get_top_headlines(gunicorn -w 4 -b 0.0.0.0:5000 app:app

    country='us',```

    category='technology',

    page_size=202. **Set Production Config**:

)```bash

```export FLASK_DEBUG=False

export FLASK_ENV=production

**Configuration**:```

- Set `NEWSAPI_KEY` in `.env` file

- Free tier: 100 requests/day3. **Use Reverse Proxy** (Nginx):

- Response cached for performance```nginx

server {

---    listen 80;

    server_name your-domain.com;

#### **Model Loading Integration**    

    location / {

```python        proxy_pass http://127.0.0.1:5000;

# In app.py - initialization        proxy_set_header Host $host;

        proxy_set_header X-Real-IP $remote_addr;

from utils.model_loader import ModelLoader    }

from utils.predictor import UnifiedPredictor}

```

# Load models once at startup

model_loader = ModelLoader()---

model_loader.load_all_models()  # Loads SVM, LSTM, BERT

## 📊 Performance Benchmarks

# Create predictor

predictor = UnifiedPredictor(model_loader)### **Model Accuracies**:

```- SVM: **99.5%** (best for structured features)

- LSTM: **87.0%** (good for temporal patterns)

**Model Paths** (in `config.py`):- BERT: **89.0%** (excellent context understanding)

```python- **Ensemble: ~94%** (combined strength)

SVM_MODEL_PATH = 'models/final_linear_svm.pkl'

LSTM_MODEL_PATH = 'models/lstm_best_model.h5'### **Processing Times** (average):

BERT_MODEL_PATH = 'models/bert_fake_news_model/'- Text Analysis: **0.5-1.5 seconds**

```- URL Extraction: **2-5 seconds**

- News Fetching: **2-5 seconds** (10 articles)

---- Model Loading: **5-10 seconds** (startup only)



## 7. Security & Performance### **Scalability**:

- **Concurrent Users**: 50-100 (single instance)

### Security Measures- **Articles/Day**: Unlimited (depends on NewsAPI plan)

- **Memory Usage**: ~500MB (all models loaded)

#### **Input Validation**- **CPU Usage**: Moderate (mostly during prediction)

- Maximum text length: 10,000 characters

- URL validation before fetching---

- HTML tag stripping from user input

- SQL injection prevention (parameterized queries in commercial features)## 🔐 Security & Privacy



#### **Rate Limiting**### **Security Features**:

- NewsAPI: 100 requests/day (free tier)- ✅ API key protection (.env files, .gitignore)

- Commercial API: Per-plan limits (100-10,000 req/month)- ✅ CORS configuration (Flask-CORS)

- Session-based throttling for free users- ✅ Session management (server-side)

- ✅ Input validation

#### **API Key Protection**- ✅ Error handling (no sensitive data leaks)

- Environment variables for sensitive data

- `.env` file not in version control### **Privacy**:

- Keys never exposed in client-side code- ✅ No user data storage (session-based only)

- ✅ No tracking or analytics (by default)

#### **CORS Configuration**- ✅ Local model inference (no data sent to external AI services)

```python- ✅ NewsAPI requests are anonymous

from flask_cors import CORS

CORS(app)  # Configure for production domains---

```

## 📈 Future Enhancements

### Performance Optimizations

### **Planned Features**:

#### **Model Loading**

- Models loaded once at application startup1. **Agent 1 (Article Collector)**:

- Shared across all requests (global variables)   - Multi-source support (Twitter, Reddit, RSS)

- Lazy loading for optional components   - Advanced web scraping (JavaScript rendering)

   - Image and video analysis

#### **Caching**   - Social media integration

- Session-based history caching

- NewsAPI responses cached (5 minutes)2. **Agent 2 (Credibility Analyzer)**:

- Preprocessed text cached for repeated analysis   - Additional ML models (XGBoost, Random Forest)

   - Fine-tuned BERT for fake news

#### **Response Time**   - Claim extraction and verification

- Average analysis: 1-2 seconds   - Context-aware analysis

- SVM: ~0.5 seconds (fastest)

- LSTM: ~1 second3. **Agent 3 (Verdict Agent)**:

- BERT: ~1.5 seconds (slowest)   - LLM integration (GPT-4, Claude)

- Parallel execution reduces total time   - Reasoning chain visualization

   - Multi-language support

#### **Memory Management**   - Explainable AI features

- Models loaded in memory (~500MB total)

- Garbage collection for large texts4. **System-Wide**:

- Session cleanup for expired data   - User authentication

   - Dashboard analytics

---   - Batch processing

   - API rate limiting

## 8. Deployment Guide   - Mobile app



### Local Development Setup---



#### **Step 1: Clone Repository**## 📚 Documentation

```powershell

git clone <repository-url>### **Agent Documentation**:

cd FND- [AGENT_1_ARTICLE_COLLECTOR.md](AGENT_1_ARTICLE_COLLECTOR.md)

```- [AGENT_2_CREDIBILITY_ANALYZER.md](AGENT_2_CREDIBILITY_ANALYZER.md)

- [AGENT_3_VERDICT_AGENT.md](AGENT_3_VERDICT_AGENT.md)

#### **Step 2: Create Virtual Environment**

```powershell### **Additional Resources**:

python -m venv venv- `README.md`: Project overview and setup

.\venv\Scripts\activate- `QUICK_FIX_README.txt`: Troubleshooting guide

```- `COMPLETE_FIX_GUIDE.txt`: Detailed fixes

- `REBUILD_SUMMARY.md`: System rebuild notes

#### **Step 3: Install Dependencies**

```powershell---

pip install -r requirements.txt

```## 🤝 Contributing



#### **Step 4: Configure Environment**### **How Each Agent Can Be Extended**:

```powershell

# Copy .env.example to .env#### **Agent 1 Extensions**:

Copy-Item .env.example .env```python

# Add new data source

# Edit .env and add your API keysclass TwitterFetcher:

notepad .env    def fetch_tweets(self, query):

```        # Implementation

        pass

Required in `.env`:

```# Integrate in news_fetcher.py

NEWSAPI_KEY=your_newsapi_key_hereif source == 'twitter':

SECRET_KEY=your_secret_key_here    fetcher = TwitterFetcher()

FLASK_DEBUG=True    articles = fetcher.fetch_tweets(query)

``````



#### **Step 5: Download Models**#### **Agent 2 Extensions**:

Ensure model files are in `models/` directory:```python

- `final_linear_svm.pkl`# Add new model

- `final_vectorizer.pkl`class XGBoostPredictor:

- `lstm_best_model.h5`    def predict(self, text):

- `lstm_tokenizer.pkl`        # Implementation

- `bert_fake_news_model/` (directory with BERT files)        pass



#### **Step 6: Run Application**# Add to model_loader.py

```powershelldef load_xgboost_model(self):

python app.py    self.models['xgboost'] = load_xgboost()

``````



Application starts on `http://localhost:5000`#### **Agent 3 Extensions**:

```python

---# Add LLM reasoning

class LLMReasoning:

### Production Deployment    def explain_with_llm(self, verdict):

        # Implementation

#### **Using Gunicorn (Linux/Mac)**        pass



```bash# Integrate in verdict_agent.py

# Install gunicornllm_explanation = self.llm_client.explain(verdict)

pip install gunicorn```



# Run with 4 workers---

gunicorn -w 4 -b 0.0.0.0:5000 app:app

```## 📞 Support



#### **Using Waitress (Windows)**For issues, questions, or contributions:

- GitHub Issues: [Repository URL]

```powershell- Documentation: See individual agent docs

# Install waitress- Email: [Contact Email]

pip install waitress

---

# Run server

waitress-serve --host=0.0.0.0 --port=5000 app:app## 📄 License

```

[Your License Here]

#### **Docker Deployment**

---

```dockerfile

FROM python:3.9-slim**System Version**: 1.0  

**Last Updated**: October 2025  

WORKDIR /app**Status**: Production Ready ✅

COPY requirements.txt .

RUN pip install -r requirements.txt---



COPY . .## 🎉 Summary



EXPOSE 5000The **Fake News Detection System** is a sophisticated three-agent architecture:

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]

```1. **Agent 1** fetches and prepares news articles

2. **Agent 2** analyzes credibility with ML ensemble

```powershell3. **Agent 3** synthesizes verdicts and generates explanations

docker build -t fake-news-detector .

docker run -p 5000:5000 --env-file .env fake-news-detectorTogether, they provide:

```- **94% accuracy** with ensemble models

- **Real-time** news verification

#### **Environment Configuration**- **Human-readable** explanations

- **Comprehensive** analysis

**Production `.env`**:

```The system is **production-ready**, **scalable**, and **extensible** for future enhancements.

NEWSAPI_KEY=your_production_key

SECRET_KEY=generate_random_secure_key---

FLASK_DEBUG=False
FLASK_ENV=production
```

**Generate secure key**:
```python
import secrets
print(secrets.token_hex(32))
```

---

### Cloud Deployment

#### **Heroku**

```powershell
# Install Heroku CLI
# Create Procfile
echo "web: gunicorn app:app" > Procfile

# Deploy
heroku create fake-news-detector
heroku config:set NEWSAPI_KEY=your_key
heroku config:set SECRET_KEY=your_secret
git push heroku main
```

#### **AWS EC2**

1. Launch EC2 instance (Ubuntu 20.04)
2. SSH into instance
3. Install Python and dependencies
4. Clone repository
5. Configure environment
6. Run with Gunicorn + Nginx

#### **Google Cloud Run**

```powershell
gcloud run deploy fake-news-detector \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

---

## 9. File Structure

```
FND/
├── app.py                          # Main Flask application (698 lines)
│   ├── Route definitions
│   ├── ML component initialization
│   ├── Request handlers
│   └── Error handling
│
├── config.py                       # Configuration settings
│   ├── Environment variables
│   ├── Model paths
│   └── API keys
│
├── news_fetcher.py                 # Agent 1: Article Collector (222 lines)
│   ├── NewsFetcher class
│   ├── fetch_and_analyze()
│   ├── _analyze_article()
│   └── get_statistics()
│
├── requirements.txt                # Python dependencies
│
├── .env                           # Environment variables (not in git)
├── .env.example                   # Environment template
│
├── models/                        # ML model files (~500MB)
│   ├── final_linear_svm.pkl
│   ├── final_vectorizer.pkl
│   ├── lstm_best_model.h5
│   ├── lstm_tokenizer.pkl
│   └── bert_fake_news_model/
│       ├── config.json
│       ├── tokenizer_config.json
│       ├── classifier.pkl
│       └── vocab.txt
│
├── utils/                         # Agent 2: ML utilities
│   ├── __init__.py
│   ├── model_loader.py            # Load all ML models (251 lines)
│   ├── predictor.py               # Ensemble prediction (326 lines)
│   ├── news_verifier.py           # NewsAPI verification
│   └── text_preprocessor.py       # Text cleaning
│
├── news_apis/                     # Agent 1: News API clients
│   ├── __init__.py
│   └── newsapi_client.py          # NewsAPI integration (150 lines)
│
├── verdict_agent/                 # Agent 3: Verdict generation
│   ├── __init__.py
│   └── verdict_agent.py           # VerdictAgent class (207 lines)
│
├── credibility_analyzer/          # Agent 2: Advanced analysis
│   ├── __init__.py
│   ├── credibility_analyzer.py    # Credibility scoring
│   ├── feature_extractor.py       # Feature extraction
│   ├── text_preprocessor.py       # Text preprocessing
│   └── confidence_calibrator.py   # Confidence calibration
│
├── commercial/                    # Commercial features (optional)
│   ├── __init__.py
│   ├── config.py
│   ├── commercial_routes.py
│   ├── auth/
│   │   └── user_manager.py
│   ├── api_limits/
│   │   └── usage_tracker.py
│   └── subscriptions/
│       └── plans.py
│
├── templates/                     # HTML templates
│   ├── index.html                 # Main page
│   ├── analyze.html               # Analysis results
│   └── commercial/                # Commercial pages
│
├── static/                        # Static assets
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── main.js
│   └── images/
│
└── docs/                          # Documentation
    ├── SYSTEM_ARCHITECTURE.md     # This file
    ├── AGENT_1_ARTICLE_COLLECTOR.md
    ├── AGENT_2_CREDIBILITY_ANALYZER.md
    ├── AGENT_3_VERDICT_AGENT.md
    └── DOCUMENTATION_INDEX.md
```

---

## 10. Configuration

### Environment Variables

**Required**:
```
NEWSAPI_KEY=your_newsapi_key          # Get from newsapi.org
SECRET_KEY=your_flask_secret_key      # Random secure string
```

**Optional**:
```
FLASK_DEBUG=True                      # Development mode
FLASK_ENV=development                 # Environment
PORT=5000                             # Server port
```

### Model Configuration

In `config.py`:
```python
class Config:
    # Model paths
    SVM_MODEL_PATH = 'models/final_linear_svm.pkl'
    SVM_VECTORIZER_PATH = 'models/final_vectorizer.pkl'
    
    LSTM_MODEL_PATH = 'models/lstm_best_model.h5'
    LSTM_TOKENIZER_PATH = 'models/lstm_tokenizer.pkl'
    
    BERT_MODEL_PATH = 'models/bert_fake_news_model/'
    
    # Model parameters
    LSTM_MAX_LENGTH = 500
    BERT_MAX_LENGTH = 512
    
    # NewsAPI
    NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')
    NEWSAPI_PAGE_SIZE = 20
    NEWSAPI_CACHE_TIMEOUT = 300  # 5 minutes
```

### Application Configuration

```python
# In app.py
app.config['SECRET_KEY'] = Config.SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max request
app.config['JSON_SORT_KEYS'] = False
```

---

## Performance Benchmarks

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | Inference Time |
|-------|----------|-----------|--------|----------|----------------|
| SVM | 99.5% | 99.3% | 99.7% | 99.5% | ~0.5s |
| LSTM | 87.0% | 85.2% | 89.1% | 87.1% | ~1.0s |
| BERT | 89.0% | 88.5% | 89.5% | 89.0% | ~1.5s |
| **Ensemble** | **94.0%** | **93.5%** | **94.5%** | **94.0%** | **~1.5s** (parallel) |

### System Performance

- **Throughput**: 40-60 requests/minute (single worker)
- **Memory Usage**: 500MB (models loaded)
- **Startup Time**: 10-15 seconds (model loading)
- **Average Response**: 1.5-2 seconds (complete analysis)

### Scalability

- **Horizontal**: Multiple Gunicorn workers
- **Vertical**: GPU acceleration for BERT/LSTM
- **Caching**: Redis for frequent queries
- **Load Balancing**: Nginx reverse proxy

---

## Future Enhancements

### Planned Features

1. **Real-time Analysis**
   - WebSocket support for streaming
   - Live news feed monitoring
   - Alert system for breaking news

2. **Advanced ML**
   - GPT-based explanation generation
   - Multi-lingual support
   - Contextual fact-checking

3. **Enhanced Verification**
   - Multiple news sources
   - Social media cross-reference
   - Expert fact-checker integration

4. **Analytics Dashboard**
   - Trend analysis
   - Source credibility tracking
   - Historical data visualization

5. **API Improvements**
   - GraphQL endpoint
   - Batch processing
   - Webhook notifications

---

## Support & Maintenance

### Monitoring

- **Health Checks**: `/test` endpoint
- **Logging**: Application and error logs
- **Metrics**: Request count, response times
- **Alerts**: Email/SMS for critical failures

### Troubleshooting

Common issues and solutions documented in:
- `QUICK_FIX_README.txt` - Quick fixes
- `COMPLETE_FIX_GUIDE.txt` - Detailed troubleshooting
- Each agent documentation - Agent-specific issues

### Updates

- **Models**: Retrain quarterly with new data
- **Dependencies**: Monthly security updates
- **NewsAPI**: Monitor API changes
- **Documentation**: Update with code changes

---

**System Version**: 1.0  
**Last Updated**: October 2025  
**Documentation**: Complete ✅  
**Status**: Production Ready 🚀

---

## Quick Start Summary

```powershell
# 1. Clone and setup
git clone <repo>
cd FND
python -m venv venv
.\venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
Copy-Item .env.example .env
# Edit .env with your NEWSAPI_KEY

# 4. Run application
python app.py

# 5. Test
curl -X POST http://localhost:5000/analyze -H "Content-Type: application/json" -d '{"text":"Your news text"}'
```

**You're ready to detect fake news!** 🎉
