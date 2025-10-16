"""
Simple Commercial Integration for Fake News Detector
Adds commercial features to existing app.py without major changes
"""

import os
import sys
import json
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

# Import your existing app configuration
from config import Config

# Initialize Flask app
app = Flask(__name__)
app.secret_key = Config.SECRET_KEY
CORS(app)

# Commercial configuration
COMMERCIAL_PLANS = {
    'starter': {
        'name': 'Starter',
        'price': 19.00,
        'analyses_limit': 500,
        'features': ['Basic text analysis', 'Email support', 'Standard confidence scores'],
        'restrictions': {'url_analysis': False, 'news_api_verification': False}
    },
    'professional': {
        'name': 'Professional', 
        'price': 99.00,
        'analyses_limit': 5000,
        'features': ['All Starter features', 'URL analysis', 'NewsAPI verification', 'API access'],
        'restrictions': {'url_analysis': True, 'news_api_verification': True}
    },
    'business': {
        'name': 'Business',
        'price': 299.00,
        'analyses_limit': 25000,
        'features': ['All Professional features', 'Batch processing', 'Custom integrations'],
        'restrictions': {'url_analysis': True, 'news_api_verification': True, 'batch_processing': True}
    },
    'enterprise': {
        'name': 'Enterprise',
        'price': 999.00,
        'analyses_limit': -1,  # Unlimited
        'features': ['All Business features', 'Unlimited analyses', 'On-premise deployment'],
        'restrictions': {'url_analysis': True, 'news_api_verification': True, 'batch_processing': True}
    }
}

# Simple user storage (in production, use a database)
USERS_FILE = 'commercial/users.json'
USAGE_FILE = 'commercial/usage.json'

def load_users():
    """Load users from file"""
    try:
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_users(users):
    """Save users to file"""
    os.makedirs('commercial', exist_ok=True)
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def load_usage():
    """Load usage data from file"""
    try:
        with open(USAGE_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_usage(usage):
    """Save usage data to file"""
    os.makedirs('commercial', exist_ok=True)
    with open(USAGE_FILE, 'w') as f:
        json.dump(usage, f, indent=2)

def check_usage_limit(user_email, user_plan):
    """Check if user has exceeded usage limits"""
    usage_data = load_usage()
    plan = COMMERCIAL_PLANS.get(user_plan, COMMERCIAL_PLANS['starter'])
    
    if user_email not in usage_data:
        usage_data[user_email] = {
            'monthly_usage': 0,
            'total_usage': 0,
            'reset_date': (datetime.now() + timedelta(days=30)).isoformat()
        }
    
    user_usage = usage_data[user_email]
    limit = plan['analyses_limit']
    
    if limit != -1 and user_usage['monthly_usage'] >= limit:
        return False, f"Usage limit exceeded. You have used {user_usage['monthly_usage']}/{limit} analyses this month."
    
    return True, "Usage within limits"

def track_usage(user_email):
    """Track user usage"""
    usage_data = load_usage()
    
    if user_email not in usage_data:
        usage_data[user_email] = {
            'monthly_usage': 0,
            'total_usage': 0,
            'reset_date': (datetime.now() + timedelta(days=30)).isoformat()
        }
    
    usage_data[user_email]['monthly_usage'] += 1
    usage_data[user_email]['total_usage'] += 1
    save_usage(usage_data)

# Import your existing ML components
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

# Store ML components in app context
app.model_loader = model_loader
app.predictor = predictor
app.news_verifier = news_verifier
app.news_fetcher = news_fetcher

# Routes

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/commercial')
def commercial_home():
    """Commercial home page"""
    return render_template('commercial.html')

@app.route('/commercial/plans')
def get_plans():
    """Get subscription plans"""
    return jsonify({'plans': COMMERCIAL_PLANS})

@app.route('/commercial/register', methods=['POST'])
def register():
    """Register new user"""
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        name = data.get('name')
        plan = data.get('plan', 'starter')
        
        if not all([email, password, name]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        users = load_users()
        
        if email in users:
            return jsonify({'error': 'User already exists'}), 400
        
        users[email] = {
            'name': name,
            'password': password,  # In production, hash this
            'plan': plan,
            'created_at': datetime.now().isoformat(),
            'is_active': True
        }
        
        save_users(users)
        
        # Set session
        session['user_email'] = email
        session['user_plan'] = plan
        
        return jsonify({'success': True, 'message': 'User registered successfully'}), 201
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/commercial/login', methods=['POST'])
def login():
    """User login"""
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        if not all([email, password]):
            return jsonify({'error': 'Email and password required'}), 400
        
        users = load_users()
        
        if email not in users:
            return jsonify({'error': 'User not found'}), 401
        
        user = users[email]
        
        if user['password'] != password:  # In production, use proper password hashing
            return jsonify({'error': 'Invalid password'}), 401
        
        if not user['is_active']:
            return jsonify({'error': 'Account is deactivated'}), 401
        
        # Set session
        session['user_email'] = email
        session['user_plan'] = user['plan']
        
        return jsonify({
            'success': True,
            'message': 'Login successful',
            'user': {
                'email': email,
                'name': user['name'],
                'plan': user['plan']
            }
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/commercial/logout', methods=['POST'])
def logout():
    """User logout"""
    session.clear()
    return jsonify({'message': 'Logged out successfully'}), 200

@app.route('/commercial/usage')
def get_usage():
    """Get user usage statistics"""
    user_email = session.get('user_email')
    user_plan = session.get('user_plan', 'starter')
    
    if not user_email:
        return jsonify({'error': 'Not authenticated'}), 401
    
    usage_data = load_usage()
    plan = COMMERCIAL_PLANS.get(user_plan, COMMERCIAL_PLANS['starter'])
    
    if user_email not in usage_data:
        usage_data[user_email] = {
            'monthly_usage': 0,
            'total_usage': 0,
            'reset_date': (datetime.now() + timedelta(days=30)).isoformat()
        }
        save_usage(usage_data)
    
    user_usage = usage_data[user_email]
    limit = plan['analyses_limit']
    
    return jsonify({
        'success': True,
        'usage': {
            'monthly_usage': user_usage['monthly_usage'],
            'total_usage': user_usage['total_usage'],
            'limit': limit,
            'remaining': limit - user_usage['monthly_usage'] if limit != -1 else 'unlimited',
            'plan': user_plan
        }
    })

@app.route('/commercial/dashboard')
def get_dashboard():
    """Get user dashboard"""
    user_email = session.get('user_email')
    user_plan = session.get('user_plan', 'starter')
    
    if not user_email:
        return jsonify({'error': 'Not authenticated'}), 401
    
    usage_data = load_usage()
    plan = COMMERCIAL_PLANS.get(user_plan, COMMERCIAL_PLANS['starter'])
    
    if user_email not in usage_data:
        usage_data[user_email] = {
            'monthly_usage': 0,
            'total_usage': 0,
            'reset_date': (datetime.now() + timedelta(days=30)).isoformat()
        }
        save_usage(usage_data)
    
    user_usage = usage_data[user_email]
    limit = plan['analyses_limit']
    
    return jsonify({
        'success': True,
        'user': {
            'email': user_email,
            'plan': user_plan
        },
        'usage': {
            'monthly_usage': user_usage['monthly_usage'],
            'total_usage': user_usage['total_usage'],
            'limit': limit,
            'remaining': limit - user_usage['monthly_usage'] if limit != -1 else 'unlimited'
        },
        'plan_info': plan
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze text for fake news with usage limits"""
    try:
        # Check authentication and usage limits
        user_email = session.get('user_email')
        user_plan = session.get('user_plan', 'starter')
        
        if user_email:
            # Check usage limits for authenticated users
            allowed, message = check_usage_limit(user_email, user_plan)
            if not allowed:
                return jsonify({
                    'error': message,
                    'upgrade_required': True,
                    'current_plan': user_plan,
                    'available_plans': list(COMMERCIAL_PLANS.keys())
                }), 402
        
        # Get original data
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        if not predictor:
            return jsonify({'error': 'ML models not loaded'}), 500
        
        # Get ML prediction
        ml_result = predictor.ensemble_predict_majority(text)
        
        # Get NewsAPI verification if available and allowed
        news_api_results = {'found': False, 'articles': [], 'error': None}
        if news_verifier and user_email:
            plan = COMMERCIAL_PLANS.get(user_plan, COMMERCIAL_PLANS['starter'])
            if plan['restrictions'].get('news_api_verification', False):
                try:
                    news_api_results = news_verifier.verify_news(text)
                except Exception as e:
                    news_api_results['error'] = str(e)
            else:
                news_api_results['error'] = 'NewsAPI verification requires Professional plan or higher'
        
        # Generate explanation
        prediction = ml_result.get('final_prediction', 'UNKNOWN')
        confidence = ml_result.get('confidence', 0)
        
        if prediction == 'FAKE':
            explanation = f"❌ ANALYSIS RESULT: High probability of misinformation (confidence: {confidence}%)."
        elif prediction == 'TRUE':
            explanation = f"✅ ANALYSIS RESULT: High probability of credible content (confidence: {confidence}%)."
        else:
            explanation = f"⚠️ ANALYSIS RESULT: Inconclusive (confidence: {confidence}%)."
        
        # Build response
        response = {
            'prediction': prediction,
            'confidence': confidence,
            'news_api_results': news_api_results,
            'individual_results': ml_result.get('individual_results', {}),
            'timestamp': datetime.now().isoformat(),
            'text': text[:100] + '...' if len(text) > 100 else text,
            'explanation': explanation,
            'commercial_info': {
                'user_plan': user_plan,
                'authenticated': bool(user_email),
                'feature_restrictions': {
                    'news_api_verification': not COMMERCIAL_PLANS.get(user_plan, COMMERCIAL_PLANS['starter'])['restrictions'].get('news_api_verification', False)
                }
            }
        }
        
        # Track usage for authenticated users
        if user_email:
            track_usage(user_email)
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze-url', methods=['POST'])
def analyze_url():
    """Analyze news article from URL with subscription limits"""
    try:
        # Check authentication
        user_email = session.get('user_email')
        user_plan = session.get('user_plan', 'starter')
        
        if not user_email:
            return jsonify({
                'error': 'URL analysis requires authentication. Please login or register.',
                'upgrade_required': True
            }), 401
        
        # Check if URL analysis is allowed for this plan
        plan = COMMERCIAL_PLANS.get(user_plan, COMMERCIAL_PLANS['starter'])
        if not plan['restrictions'].get('url_analysis', False):
            return jsonify({
                'error': 'URL analysis requires Professional plan or higher',
                'upgrade_required': True,
                'current_plan': user_plan,
                'required_plan': 'professional'
            }), 402
        
        # Check usage limits
        allowed, message = check_usage_limit(user_email, user_plan)
        if not allowed:
            return jsonify({
                'error': message,
                'upgrade_required': True
            }), 402
        
        # Get original data
        data = request.get_json()
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({'error': 'No URL provided'}), 400
        
        # Import URL extraction function from your original app
        from app import extract_article_content
        
        # Extract content from URL
        try:
            article_content = extract_article_content(url)
            if not article_content:
                return jsonify({'error': 'Could not extract content from URL'}), 400
        except Exception as e:
            return jsonify({'error': f'Failed to extract content: {str(e)}'}), 400
        
        # Get ML prediction
        if not predictor:
            return jsonify({'error': 'ML models not loaded'}), 500
        
        ml_result = predictor.ensemble_predict_majority(article_content['text'])
        
        # Get NewsAPI verification if allowed
        news_api_results = {'found': False, 'articles': [], 'error': None}
        if news_verifier and plan['restrictions'].get('news_api_verification', False):
            try:
                news_api_results = news_verifier.verify_news(article_content['text'])
            except Exception as e:
                news_api_results['error'] = str(e)
        else:
            news_api_results['error'] = 'NewsAPI verification requires Professional plan or higher'
        
        # Generate explanation
        prediction = ml_result.get('final_prediction', 'UNKNOWN')
        confidence = ml_result.get('confidence', 0)
        
        if prediction == 'FAKE':
            explanation = f"❌ ANALYSIS RESULT: High probability of misinformation (confidence: {confidence}%)."
        elif prediction == 'TRUE':
            explanation = f"✅ ANALYSIS RESULT: High probability of credible content (confidence: {confidence}%)."
        else:
            explanation = f"⚠️ ANALYSIS RESULT: Inconclusive (confidence: {confidence}%)."
        
        # Build response
        response = {
            'prediction': prediction,
            'confidence': confidence,
            'news_api_results': news_api_results,
            'individual_results': ml_result.get('individual_results', {}),
            'timestamp': datetime.now().isoformat(),
            'url': url,
            'article_title': article_content.get('title', 'Unknown Title'),
            'article_source': article_content.get('source', 'Unknown Source'),
            'article_text': article_content['text'][:200] + '...' if len(article_content['text']) > 200 else article_content['text'],
            'explanation': explanation,
            'commercial_info': {
                'user_plan': user_plan,
                'feature_restrictions': {
                    'news_api_verification': not plan['restrictions'].get('news_api_verification', False)
                }
            }
        }
        
        # Track usage
        track_usage(user_email)
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/fetch-news', methods=['POST'])
def fetch_news():
    """Fetch latest news with authentication requirement"""
    try:
        # Check authentication
        user_email = session.get('user_email')
        user_plan = session.get('user_plan', 'starter')
        
        if not user_email:
            return jsonify({
                'error': 'News fetching requires authentication. Please login or register.',
                'upgrade_required': True
            }), 401
        
        # Check usage limits
        allowed, message = check_usage_limit(user_email, user_plan)
        if not allowed:
            return jsonify({
                'error': message,
                'upgrade_required': True
            }), 402
        
        # Get original data
        data = request.get_json()
        country = data.get('country', 'us')
        category = data.get('category', 'general')
        page_size = data.get('page_size', 10)
        
        # Limit page size based on plan
        plan = COMMERCIAL_PLANS.get(user_plan, COMMERCIAL_PLANS['starter'])
        if user_plan == 'starter':
            page_size = min(page_size, 5)
        elif user_plan == 'professional':
            page_size = min(page_size, 10)
        elif user_plan == 'business':
            page_size = min(page_size, 25)
        # Enterprise has no limit
        
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
                }
            ]
            return jsonify({'articles': mock_articles})
        
        articles = news_fetcher.fetch_and_analyze(
            country=country,
            category=category,
            page_size=page_size
        )
        
        # Track usage
        track_usage(user_email)
        
        return jsonify({'articles': articles})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history', methods=['GET'])
def get_history():
    """Get analysis history"""
    user_email = session.get('user_email')
    
    if user_email:
        # For authenticated users, return empty for now (can be enhanced)
        history = []
    else:
        # For anonymous users, use session history
        history = session.get('history', [])
    
    return jsonify({'history': history})

@app.route('/clear-history', methods=['POST'])
def clear_history():
    """Clear analysis history"""
    session.clear()
    return jsonify({'message': 'History cleared'})

if __name__ == '__main__':
    print("🚀 Starting Commercial Fake News Detector...")
    print("💰 Subscription Plans Available:")
    print("   📦 Starter: $19/month - 500 analyses")
    print("   💼 Professional: $99/month - 5,000 analyses + URL analysis")
    print("   🏢 Business: $299/month - 25,000 analyses + batch processing")
    print("   🏛️ Enterprise: $999/month - Unlimited analyses")
    
    # Initialize ML components
    if initialize_ml_components():
        print("✅ Application ready!")
    else:
        print("⚠️ Failed to initialize ML components")
        print("Running in limited mode...")
    
    # Print registered routes
    print("\n📋 Available Routes:")
    print("   🏠 / - Main application")
    print("   💰 /commercial - Commercial features")
    print("   🔍 /analyze - Text analysis (with usage limits)")
    print("   🌐 /analyze-url - URL analysis (Professional+)")
    print("   📰 /fetch-news - News fetching (authenticated)")
    print("   📊 /commercial/dashboard - User dashboard")
    print("   💳 /commercial/plans - Subscription plans")
    print("   🔐 /commercial/login - User login")
    print("   📝 /commercial/register - User registration")
    
    # Run the application
    print(f"\n🌐 Starting server on {Config.FLASK_HOST}:{Config.FLASK_PORT}")
    print("🔗 Access the application at: http://localhost:5000")
    print("💡 Commercial features are now integrated!")
    
    app.run(
        debug=Config.FLASK_DEBUG,
        host=Config.FLASK_HOST,
        port=Config.FLASK_PORT,
        use_reloader=False,
        threaded=True
    )
