"""
Flask Fake News Detector Application
Clean implementation with all three ML models and NewsAPI integration
"""

import os
import sys
import json
import time
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, redirect
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

# Global variables for commercial components
user_manager = None
usage_tracker = None

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

def initialize_ml_components():
    """Initialize ML models and components"""
    global model_loader, predictor, news_verifier, news_fetcher, user_manager, usage_tracker
    
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
        
        # Initialize commercial components
        try:
            from commercial.auth.user_manager import UserManager
            from commercial.api_limits.usage_tracker import UsageTracker
            user_manager = UserManager()
            usage_tracker = UsageTracker()
            print("Commercial components initialized")
        except Exception as e:
            print(f"WARNING: Commercial components not available: {e}")
            user_manager = None
            usage_tracker = None
        
        print("ML components initialized successfully!")
        return True
        
    except Exception as e:
        print(f"Error initializing ML components: {e}")
        return False

def add_to_history(result):
    """Add analysis result to session history"""
    if 'history' not in session:
        session['history'] = []
    session['history'].append(result)
    session.modified = True

def extract_article_content(url):
    """Extract article content from URL using requests and BeautifulSoup"""
    try:
        import requests
        from bs4 import BeautifulSoup
        import re
        from urllib.parse import urlparse
        
        # Set headers to mimic a real browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Make request to URL
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Try to find article title
        title = None
        title_selectors = [
            'h1', 'title', '[class*="title"]', '[class*="headline"]',
            'meta[property="og:title"]', 'meta[name="title"]'
        ]
        
        for selector in title_selectors:
            if selector.startswith('meta'):
                meta_tag = soup.select_one(selector)
                if meta_tag and meta_tag.get('content'):
                    title = meta_tag.get('content').strip()
                    break
            else:
                title_elem = soup.select_one(selector)
                if title_elem and title_elem.get_text().strip():
                    title = title_elem.get_text().strip()
                    break
        
        # Try to find article content
        content = None
        content_selectors = [
            'article', '[class*="article"]', '[class*="content"]', '[class*="post"]',
            '[class*="story"]', 'main', '.entry-content', '.post-content'
        ]
        
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                # Get all paragraphs
                paragraphs = content_elem.find_all(['p', 'div'])
                if paragraphs:
                    content = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
                    break
        
        # Fallback: get all paragraph text
        if not content:
            paragraphs = soup.find_all('p')
            if paragraphs:
                content = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        
        # Clean up content
        if content:
            # Remove extra whitespace
            content = re.sub(r'\s+', ' ', content).strip()
            # Remove very short sentences (likely navigation/ads)
            sentences = content.split('.')
            content = '. '.join([s.strip() for s in sentences if len(s.strip()) > 20])
        
        # Extract source/domain
        domain = urlparse(url).netloc
        source = domain.replace('www.', '').split('.')[0].title()
        
        if not content or len(content) < 50:
            return None
        
        return {
            'title': title or 'Unknown Title',
            'text': content,
            'source': source,
            'url': url
        }
        
    except Exception as e:
        print(f"Error extracting content from {url}: {e}")
        return None

# Routes
@app.route('/')
def index():
    """Serve the home page as the default landing page"""
    return render_template('home.html')

@app.route('/detector')
def detector():
    """Serve the main detector page"""
    return render_template('index.html')

@app.route('/home')
def home():
    """Serve the home page with news banners and feedback"""
    return render_template('home.html')

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

@app.route('/analyze-url', methods=['POST'])
def analyze_url():
    """Analyze news article from URL for fake news detection"""
    try:
        print("=== ANALYZE URL REQUEST START ===")
        data = request.get_json()
        url = data.get('url', '').strip()
        print(f"URL to analyze: {url}")

        if not url:
            print("No URL provided")
            return jsonify({'error': 'No URL provided'}), 400

        if not predictor:
            return jsonify({'error': 'ML models not loaded'}), 500
        
        # Extract content from URL
        try:
            article_content = extract_article_content(url)
            if not article_content:
                return jsonify({'error': 'Could not extract content from URL'}), 400
        except Exception as e:
            return jsonify({'error': f'Failed to extract content: {str(e)}'}), 400
        
        # Get ML prediction on extracted content
        ml_result = predictor.ensemble_predict_majority(article_content['text'])
        
        # Get NewsAPI verification if available
        news_api_results = {'found': False, 'articles': [], 'error': None}
        if news_verifier:
            try:
                news_api_results = news_verifier.verify_news(article_content['text'])
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
            'url': url,
            'article_title': article_content.get('title', 'Unknown Title'),
            'article_source': article_content.get('source', 'Unknown Source'),
            'article_text': article_content['text'][:200] + '...' if len(article_content['text']) > 200 else article_content['text'],
            'explanation': explanation
        }
        
        # Make entire response JSON-safe
        response = _make_json_safe(response)
        
        # Store in history
        add_to_history(response)
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in analyze_url: {e}")
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

        data = request.get_json(silent=True) or {}
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
    return jsonify({'history': _make_json_safe(history)})

@app.route('/clear-history', methods=['POST'])
def clear_history():
    """Clear analysis history"""
    session.clear()  # Clear entire session to remove any problematic data
    return jsonify({'message': 'History cleared'})

# Commercial Routes
@app.route('/pricing')
def pricing():
    """Display pricing plans page"""
    return render_template('pricing.html')

@app.route('/payment')
def payment():
    """Display payment page"""
    plan = request.args.get('plan', 'professional')
    return render_template('payment.html', plan=plan)

@app.route('/login')
def login_page():
    """Display login page"""
    # If user is already logged in, redirect to home
    if 'user_email' in session:
        return redirect('/')
    return render_template('login.html')

@app.route('/register')
def register_page():
    """Display registration page"""
    # If user is already logged in, redirect to home
    if 'user_email' in session:
        return redirect('/')
    return render_template('register.html')

@app.route('/dashboard')
def dashboard_page():
    """Display user dashboard"""
    if 'user_email' not in session:
        return redirect('/login')
    return render_template('commercial.html', page='dashboard')

# Commercial API endpoints
@app.route('/api/auth/login', methods=['POST'])
def api_login():
    """Handle user login"""
    if not user_manager:
        return jsonify({'success': False, 'message': 'User management not available'}), 500
    
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    
    result = user_manager.authenticate_user(email, password)
    if result['success']:
        session['user_email'] = email
        session['user_plan'] = result['user']['plan']
        session['user_name'] = result['user']['name']
        return jsonify(result)
    return jsonify(result), 401

@app.route('/api/auth/register', methods=['POST'])
def api_register():
    """Handle user registration"""
    if not user_manager:
        return jsonify({'success': False, 'message': 'User management not available'}), 500
    
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    name = data.get('name')
    plan = data.get('plan', 'starter')
    
    result = user_manager.register_user(email, password, name, plan)
    if result['success']:
        session['user_email'] = email
        session['user_plan'] = plan
        session['user_name'] = name
        return jsonify(result)
    return jsonify(result), 400

@app.route('/api/auth/logout', methods=['POST'])
def api_logout():
    """Handle user logout"""
    session.clear()
    return jsonify({'success': True, 'message': 'Logged out successfully'})

@app.route('/api/payment/process', methods=['POST'])
def process_payment():
    """Mock payment processing (no actual Stripe integration)"""
    if not user_manager:
        return jsonify({'success': False, 'message': 'User management not available'}), 500
    
    data = request.get_json()
    plan = data.get('plan', 'professional')
    
    # Mock successful payment
    if 'user_email' in session:
        # Update user plan
        user_manager.users[session['user_email']]['plan'] = plan
        user_manager._save_users()
        session['user_plan'] = plan
        
        return jsonify({
            'success': True,
            'message': 'Payment processed successfully (mock)',
            'plan': plan
        })
    
    return jsonify({'success': False, 'message': 'User not logged in'}), 401

@app.route('/api/user/usage', methods=['GET'])
def get_user_usage():
    """Get user usage statistics"""
    if 'user_email' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    if not usage_tracker:
        return jsonify({'error': 'Usage tracking not available'}), 500
    
    user_email = session['user_email']
    usage_stats = usage_tracker.get_usage_stats(user_email)
    
    # Get plan info
    from commercial.subscriptions.plans import SubscriptionPlans
    plan_info = SubscriptionPlans.get_plan(session.get('user_plan', 'starter'))
    
    return jsonify({
        'usage': usage_stats,
        'plan': plan_info
    })

@app.route('/api/feedback', methods=['GET'])
def get_feedback():
    """Get user feedback for display on home page"""
    try:
        # Load feedback from commercial/feedback.json
        feedback_file = "commercial/feedback.json"
        try:
            with open(feedback_file, 'r') as f:
                feedback_data = json.load(f)
        except FileNotFoundError:
            feedback_data = []
        
        # Filter approved feedback and sort by submission date (newest first)
        approved_feedback = [item for item in feedback_data if item.get('status') == 'approved']
        
        # Sort by submission date (newest first) and get latest 3
        approved_feedback.sort(key=lambda x: x.get('submitted_at', ''), reverse=True)
        latest_feedback = approved_feedback[:3]
        
        return jsonify({
            'success': True,
            'feedback': latest_feedback,
            'total_approved': len(approved_feedback)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'feedback': []
        })

@app.route('/submit-feedback', methods=['POST'])
def submit_feedback():
    """Handle feedback submission from users"""
    try:
        data = request.get_json()
        
        # Extract feedback data
        name = data.get('name', '').strip()
        email = data.get('email', '').strip()
        feedback_text = data.get('feedback', '').strip()
        rating = data.get('rating', 5)
        
        # Validate required fields
        if not name or not email or not feedback_text:
            return jsonify({
                'success': False,
                'message': 'Name, email, and feedback are required'
            }), 400
        
        # Create feedback entry
        feedback_entry = {
            'id': f"feedback_{int(time.time())}",
            'name': name,
            'email': email,
            'feedback': feedback_text,
            'rating': int(rating),
            'status': 'approved',  # Auto-approve for demo purposes
            'submitted_at': datetime.now().isoformat(),
            'user_plan': 'free'  # Default for feedback submissions
        }
        
        # Load existing feedback
        feedback_file = "commercial/feedback.json"
        try:
            with open(feedback_file, 'r') as f:
                feedback_data = json.load(f)
        except FileNotFoundError:
            feedback_data = []
        
        # Add new feedback
        feedback_data.append(feedback_entry)
        
        # Save updated feedback
        with open(feedback_file, 'w') as f:
            json.dump(feedback_data, f, indent=2)
        
        return jsonify({
            'success': True,
            'message': 'Thank you for your feedback! It will appear on our home page immediately.',
            'feedback_id': feedback_entry['id']
        })
        
    except Exception as e:
        print(f"Error submitting feedback: {e}")
        return jsonify({
            'success': False,
            'message': 'Failed to submit feedback. Please try again.'
        }), 500

def generate_explanation(ml_result, news_api_results):
    """Generate explanation based on ML prediction and verification"""
    prediction = ml_result.get('final_prediction', 'UNKNOWN')
    confidence = ml_result.get('confidence', 0)
    
    # Start with ML analysis explanation
    explanation_parts = []
    
    # Add ML prediction explanation
    if prediction == 'FAKE':
        explanation_parts.append(f"🤖 AI ANALYSIS: High probability of misinformation detected (confidence: {confidence:.1f}%)")
        explanation_parts.append("The text contains patterns commonly found in fabricated or satirical content.")
    elif prediction == 'TRUE':
        explanation_parts.append(f"🤖 AI ANALYSIS: High probability of credible content (confidence: {confidence:.1f}%)")
        explanation_parts.append("The text patterns are consistent with factual reporting.")
    else:
        explanation_parts.append(f"🤖 AI ANALYSIS: Inconclusive result (confidence: {confidence:.1f}%)")
        explanation_parts.append("Additional verification may be needed.")
    
    # Add individual model results if available
    individual_results = ml_result.get('individual_results', {})
    if individual_results:
        explanation_parts.append("\n📊 INDIVIDUAL MODEL RESULTS:")
        for model_name, result in individual_results.items():
            if isinstance(result, dict) and 'prediction' in result:
                model_pred = result.get('prediction', 'UNKNOWN')
                model_conf = result.get('confidence', 0)
                explanation_parts.append(f"• {model_name}: {model_pred} ({model_conf:.1f}%)")
    
    # Check if NewsAPI found matching articles
    if news_api_results.get('found_online') and news_api_results.get('articles'):
        articles = news_api_results.get('articles', [])
        best_match = news_api_results.get('best_match', {})
        similarity = best_match.get('similarity_score', 0)
        
        explanation_parts.append(f"\n✅ ONLINE VERIFICATION: Found {len(articles)} similar article(s) from trusted sources.")
        
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
                article_url = article.get('url', '#')
                explanation_parts.append(f"{i}. {article_title} ({article_source}) - {article_url}")
    
    else:
        # No online verification available - silently skip the warning
        if news_api_results.get('error'):
            explanation_parts.append(f"\n⚠️ ONLINE VERIFICATION: Unable to verify online - {news_api_results['error']}")
        # Removed the "No matching articles found" warning message
    
    return "\n".join(explanation_parts)

# Initialize the application
if __name__ == '__main__':
    print("Starting Flask Fake News Detector...")
    
    # Initialize ML components
    if initialize_ml_components():
        print("Application ready!")
        print("Registered routes:")
        for rule in app.url_map.iter_rules():
            print(f"  {rule.rule}")
        
        # Start the server
        app.run(
            host=Config.FLASK_HOST,
            port=Config.FLASK_PORT,
            debug=Config.FLASK_DEBUG
        )
    else:
        print("Failed to initialize ML components. Exiting.")
        sys.exit(1)
