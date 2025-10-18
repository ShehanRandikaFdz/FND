# Fake News Detector - AI Agent Instructions

## Architecture Overview

This is a **dual-mode Flask application** combining a free fake news detection service with a commercial SaaS platform. The system uses **ensemble ML** (SVM, LSTM, BERT) with majority voting and external verification through NewsAPI.

### Key Components

- **ML Pipeline**: Three models (`utils/`) loaded once at startup via `ModelLoader`, predictions unified through `UnifiedPredictor.ensemble_predict_majority()`
- **Commercial Layer**: Subscription-based features (`commercial/`) with usage tracking, authentication, and Stripe integration - initialized optionally alongside core ML components
- **Verification**: NewsAPI integration (`utils/news_verifier.py`) with semantic similarity matching using sentence-transformers
- **Advanced Analysis**: `credibility_analyzer/` and `verdict_agent/` provide multi-factor credibility scoring and consensus-based verdicts

## Critical Workflows

### Running the Application

```powershell
# Setup (first time)
pip install -r requirements.txt
# Copy .env.example to .env and add your NEWSAPI_KEY

# Run (loads all models on startup)
python app.py  # Listens on 0.0.0.0:5000 by default
```

**Important**: Models load at startup in `initialize_ml_components()`. If any model fails, the app continues with available models - check console output for status.

### Testing Analysis Endpoints

```powershell
# Basic text analysis (majority voting across all loaded models)
curl -X POST http://localhost:5000/analyze -H "Content-Type: application/json" -d '{"text":"Your news text here"}'

# URL analysis (extracts content then analyzes)
curl -X POST http://localhost:5000/analyze-url -H "Content-Type: application/json" -d '{"url":"https://example.com/article"}'

# Fetch news (requires NEWSAPI_KEY)
curl -X POST http://localhost:5000/fetch-news -H "Content-Type: application/json" -d '{"country":"us","category":"general"}'
```

## Project-Specific Patterns

### 1. Ensemble Prediction Pattern

**All analysis goes through `UnifiedPredictor.ensemble_predict_majority()`** which:
- Collects predictions from all loaded models (SVM/LSTM/BERT)
- Uses majority voting to determine final verdict
- Handles missing models gracefully
- Returns JSON-safe results (numpy types converted via `_make_json_safe()`)

**When adding features**: Always call `predictor.ensemble_predict_majority(text)` rather than individual model methods.

### 2. Lazy Model Loading

Models are loaded once globally:
```python
# In app.py
model_loader = None  # Global
predictor = None     # Global

def initialize_ml_components():
    global model_loader, predictor
    model_loader = ModelLoader()
    model_loader.load_all_models()  # Loads SVM, LSTM, BERT
    predictor = UnifiedPredictor(model_loader)
```

**Never** recreate these objects per-request. Reference the globals.

### 3. Commercial/Free Hybrid

The app runs both modes simultaneously:
- **Free routes**: `/analyze`, `/analyze-url`, `/fetch-news` - no auth required
- **Commercial routes**: `/commercial/*` - authentication, usage limits, subscriptions

Commercial components initialize separately and fail gracefully if unavailable. Check `user_manager` and `usage_tracker` globals before using.

### 4. JSON Safety

Flask routes MUST return JSON-safe types. Always use:
```python
from app import _make_json_safe  # or define locally
response = _make_json_safe(ml_result)  # Converts numpy types
return jsonify(response)
```

**Why**: ML models return numpy types (`np.float64`, `np.int64`) which cause `Object of type float64 is not JSON serializable` errors.

### 5. NewsAPI Verification Pattern

```python
# Check if available before using
if news_verifier:
    try:
        news_api_results = news_verifier.verify_news(text)
    except Exception as e:
        news_api_results = {'found': False, 'error': str(e)}
```

NewsAPI has rate limits (100 req/day free tier). Handle failures gracefully.

## Configuration

All config in `config.py` using environment variables:
- `NEWSAPI_KEY`: Required for news verification and fetching
- `SECRET_KEY`: Flask session security (change in production!)
- `FLASK_DEBUG`: Set to `False` in production
- Model paths: Default to `models/` directory

**Never commit API keys**. Use `.env` file (see `.env.example`).

## Common Gotchas

1. **TensorFlow/Transformers Compatibility**: LSTM/BERT models may fail to load due to library conflicts. App continues with SVM only - check `model_loader.model_status` dict
2. **Session Storage**: Analysis history stored in Flask sessions (not DB). Clear with `/clear-history`
3. **URL Extraction**: `extract_article_content()` uses BeautifulSoup with heuristic selectors - may fail on complex sites
4. **Windows Paths**: Project uses forward slashes in code but Windows paths in file system. `os.path.join()` handles this
5. **Commercial JSON Files**: `commercial/users.json`, `commercial/feedback.json` must exist for commercial features to work

## File Organization

```
app.py                          # Main Flask app, all routes, model initialization
config.py                       # Environment-based configuration
utils/                          # Core ML pipeline
  ├── model_loader.py          # Loads SVM/LSTM/BERT models
  ├── predictor.py             # UnifiedPredictor with ensemble logic
  ├── news_verifier.py         # NewsAPI integration + similarity
  └── text_preprocessor.py     # Basic text cleaning
credibility_analyzer/          # Advanced multi-factor analysis
verdict_agent/                 # Multi-agent consensus system
commercial/                    # SaaS features (auth, subscriptions, usage tracking)
  ├── auth/user_manager.py    # User registration/authentication
  ├── api_limits/usage_tracker.py  # Rate limiting per plan
  └── subscriptions/plans.py   # Tier definitions (Starter/Pro/Business/Enterprise)
models/                        # Pre-trained model files (not in git)
templates/                     # Jinja2 HTML templates
static/js/main.js              # Frontend logic
```

## Adding New Features

### New Analysis Endpoint
1. Add route to `app.py`
2. Call `predictor.ensemble_predict_majority(text)`
3. Optionally verify with `news_verifier.verify_news(text)`
4. Use `_make_json_safe()` before `jsonify()`
5. Add to session history with `add_to_history(result)`

### New ML Model
1. Add to `utils/model_loader.py` (new `load_X_model()` method)
2. Add to `utils/predictor.py` (new `predict_X()` method)
3. Update `ensemble_predict_majority()` to include new model votes
4. Update `Config` class with model paths

### Commercial Feature
1. Add to `commercial/subscriptions/plans.py` PLANS dict
2. Update `SubscriptionPlans.is_feature_allowed()`
3. Check access in routes: `check_feature_access(user_plan, 'feature_name')`
4. Track usage: `usage_tracker.track_analysis(user_email, 'analysis_type')`

## Dependencies

**Critical libraries** (see `requirements.txt`):
- `scikit-learn>=1.4.0` - SVM model
- `tensorflow>=2.13.0` - LSTM model (optional)
- `transformers>=4.30.0` - BERT model (optional)
- `sentence-transformers>=2.2.0` - Semantic similarity
- `newsapi-python==0.2.7` - News verification
- `Flask==3.0.0` - Web framework
- `beautifulsoup4==4.12.2` - URL content extraction

Install all: `pip install -r requirements.txt`

## Testing Strategy

No formal test suite currently. Manual testing via:
1. Start app: `python app.py`
2. Check console for model load status
3. Test `/analyze` with sample text
4. Test `/analyze-url` with real URL
5. Test `/fetch-news` (requires API key)

Commercial features tested via web UI at `http://localhost:5000/commercial/login`

---

**Remember**: This is a hybrid free/commercial platform with graceful degradation. Always check if optional components (`news_verifier`, `user_manager`, models) are loaded before using them.
