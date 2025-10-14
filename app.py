"""
Flask Fake News Detector Application
Clean implementation with all three ML models and NewsAPI integration
"""

import os
import sys
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from config import Config

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

# Initialize Flask app
app = Flask(__name__)
app.secret_key = Config.SECRET_KEY
CORS(app)

# Global variables for ML components
model_loader = None
predictor = None
news_verifier = None
news_fetcher = None

def initialize_ml_components():
    """Initialize ML models and components"""
    global model_loader, predictor, news_verifier, news_fetcher
    
    try:
        print("Initializing ML components...")
        
        # Initialize model loader
        from utils.model_loader import ModelLoader
        model_loader = ModelLoader()
        
        # Load all models
        success = model_loader.load_all_models()
        
        if success:
            # Initialize predictor
            from utils.predictor import UnifiedPredictor
            predictor = UnifiedPredictor(model_loader)
            print("ML models and predictor initialized")
        else:
            print("WARNING: Some models failed to load, but predictor will use available models")
            from utils.predictor import UnifiedPredictor
            predictor = UnifiedPredictor(model_loader)
        
        # Initialize news verifier
        try:
            from utils.news_verifier import NewsVerifier
            news_verifier = NewsVerifier()
            print("News verifier initialized")
        except Exception as e:
            print(f"WARNING: News verifier not available: {e}")
            news_verifier = None
        
        # Initialize news fetcher
        try:
            from news_fetcher import NewsFetcher
            news_fetcher = NewsFetcher()
            print("News fetcher initialized")
        except Exception as e:
            print(f"WARNING: News fetcher not available: {e}")
            news_fetcher = None
        
        print("ML components initialized successfully!")
        return True
        
    except Exception as e:
        print(f"Error initializing ML components: {e}")
        return False

def _make_json_safe(obj):
    """Convert numpy types to JSON-safe Python types"""
    import numpy as np
    
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_safe(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    elif hasattr(obj, 'tolist'):  # numpy array
        return obj.tolist()
    else:
        return obj

def add_to_history(result):
    """Add analysis result to session history"""
    if 'history' not in session:
        session['history'] = []
    session['history'].append(result)
    session.modified = True

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze text for fake news"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        if not predictor:
            return jsonify({'error': 'ML models not loaded'}), 500
        
        # Get ML prediction
        ml_result = predictor.ensemble_predict_majority(text)
        
        # Get NewsAPI verification if available
        news_api_results = {'found': False, 'articles': [], 'error': None}
        if news_verifier:
            try:
                news_api_results = news_verifier.verify_news(text)
            except Exception as e:
                news_api_results['error'] = str(e)
        
        # Generate explanation
        explanation = generate_explanation(ml_result, news_api_results)
        
        # Build response with JSON-safe types
        response = {
            'prediction': ml_result.get('final_prediction', 'UNKNOWN'),
            'confidence': ml_result.get('confidence', 0),
            'news_api_results': news_api_results,
            'individual_results': ml_result.get('individual_results', {}),
            'timestamp': datetime.now().isoformat(),
            'text': text[:100] + '...' if len(text) > 100 else text,
            'explanation': explanation
        }
        
        # Make entire response JSON-safe
        response = _make_json_safe(response)
        
        # Store in history
        add_to_history(response)
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in analyze: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/fetch-news', methods=['POST'])
def fetch_news():
    """Fetch latest news from NewsAPI"""
    try:
        if not news_fetcher:
            # Return mock data if news fetcher not available
            mock_articles = [
                {
                    'title': 'Sample News Article 1',
                    'description': 'This is a sample news article for testing purposes.',
                    'url': 'https://example.com/article1',
                    'source': 'Sample News',
                    'published_at': '2024-10-13',
                    'credibility_score': 0.8,
                    'prediction': 'TRUE',
                    'confidence': 85
                },
                {
                    'title': 'Sample News Article 2', 
                    'description': 'Another sample article for demonstration.',
                    'url': 'https://example.com/article2',
                    'source': 'Demo News',
                    'published_at': '2024-10-13',
                    'credibility_score': 0.3,
                    'prediction': 'FAKE',
                    'confidence': 75
                }
            ]
            return jsonify({'articles': mock_articles})
        
        data = request.get_json()
        country = data.get('country', 'us')
        category = data.get('category', 'general')
        page_size = data.get('page_size', 10)
        
        articles = news_fetcher.fetch_and_analyze(
            country=country,
            category=category,
            page_size=page_size
        )
        
        return jsonify({'articles': articles})
        
    except Exception as e:
        print(f"Error fetching news: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/history', methods=['GET'])
def get_history():
    """Get analysis history"""
    history = session.get('history', [])
    return jsonify({'history': history})

@app.route('/clear-history', methods=['POST'])
def clear_history():
    """Clear analysis history"""
    session.clear()  # Clear entire session to remove any problematic data
    return jsonify({'message': 'History cleared'})

def generate_explanation(ml_result, news_api_results):
    """Generate explanation based on ML prediction and verification"""
    prediction = ml_result.get('final_prediction', 'UNKNOWN')
    confidence = ml_result.get('confidence', 0)
    
    # Check if NewsAPI found matching articles
    if news_api_results.get('found_online') and news_api_results.get('articles'):
        articles = news_api_results.get('articles', [])
        best_match = news_api_results.get('best_match', {})
        similarity = best_match.get('similarity_score', 0)
        
        # Create detailed explanation with found articles
        explanation_parts = []
        
        # Main verification result
        explanation_parts.append(f"✅ ONLINE VERIFICATION: Found {len(articles)} similar article(s) from trusted sources.")
        
        # Best match details
        if best_match:
            title = best_match.get('title', 'Unknown Title')
            source = best_match.get('source', {}).get('name', 'Unknown Source')
            published_at = best_match.get('publishedAt', 'Unknown Date')
            url = best_match.get('url', '#')
            
            explanation_parts.append(f"📰 BEST MATCH: '{title}'")
            explanation_parts.append(f"🏢 SOURCE: {source}")
            explanation_parts.append(f"📅 PUBLISHED: {published_at}")
            explanation_parts.append(f"🎯 SIMILARITY: {similarity:.1%}")
            explanation_parts.append(f"🔗 READ MORE: {url}")
        
        # Additional articles if any
        if len(articles) > 1:
            explanation_parts.append(f"\n📚 OTHER MATCHES ({len(articles)-1} more):")
            for i, article in enumerate(articles[1:3], 1):  # Show up to 2 additional articles
                article_title = article.get('title', 'Unknown Title')
                article_source = article.get('source', {}).get('name', 'Unknown Source')
                explanation_parts.append(f"  {i}. {article_title} ({article_source})")
        
        return "\n".join(explanation_parts)
    
    # Fallback to ML-based explanation
    if prediction == 'FAKE':
        return f"❌ ANALYSIS RESULT: High probability of misinformation (confidence: {confidence}%).\n\nText patterns suggest sensationalism or lack of factual basis. No matching articles found in trusted sources."
    elif prediction == 'TRUE':
        return f"✅ ANALYSIS RESULT: High probability of credible content (confidence: {confidence}%).\n\nText patterns are consistent with factual reporting. No matching articles found in trusted sources for verification."
    else:
        return f"⚠️ ANALYSIS RESULT: Inconclusive (confidence: {confidence}%).\n\nAdditional verification may be needed. No matching articles found in trusted sources."

if __name__ == '__main__':
    print("Starting Flask Fake News Detector...")
    
    # Initialize ML components
    if initialize_ml_components():
        print("Application ready!")
    else:
        print("Failed to initialize ML components")
        print("Running in limited mode...")
    
    # Print registered routes
    print("Registered routes:")
    for rule in app.url_map.iter_rules():
        print(f"  {rule}")
    
    # Run the application
    print(f"Starting server on {Config.FLASK_HOST}:{Config.FLASK_PORT}")
    app.run(
        debug=Config.FLASK_DEBUG,
        host=Config.FLASK_HOST,
        port=Config.FLASK_PORT,
        use_reloader=False,
        threaded=True
    )