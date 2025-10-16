"""
Simple Commercial Fake News Detector
Minimal working version with commercial features
"""

from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import json
import os
from datetime import datetime, timedelta

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'commercial-secret-key-change-in-production'
CORS(app)

# Enhanced analysis function
def enhanced_analysis(text):
    """Enhanced analysis using multiple heuristics and patterns"""
    text_lower = text.lower()
    
    # Fake news indicators (more comprehensive)
    fake_indicators = [
        'fake', 'false', 'misleading', 'hoax', 'conspiracy', 'clickbait',
        'unverified', 'rumor', 'allegedly', 'supposedly', 'claims without evidence',
        'viral', 'shocking', 'you won\'t believe', 'doctors hate this',
        'one weird trick', 'this one simple trick', 'miracle cure',
        'instant results', 'guaranteed', 'secret', 'hidden truth'
    ]
    
    # Real news indicators
    real_indicators = [
        'according to', 'study shows', 'research indicates', 'official',
        'government', 'university', 'scientists', 'experts say',
        'peer-reviewed', 'published in', 'journal', 'study published',
        'data shows', 'statistics', 'survey', 'poll', 'report',
        'confirmed', 'verified', 'fact-checked', 'reliable source'
    ]
    
    # Emotional manipulation indicators
    emotional_indicators = [
        'urgent', 'breaking', 'shocking', 'outrageous', 'scandal',
        'exposed', 'revealed', 'leaked', 'insider', 'whistleblower',
        'cover-up', 'conspiracy', 'plot', 'scheme'
    ]
    
    # Calculate scores
    fake_score = sum(1 for indicator in fake_indicators if indicator in text_lower)
    real_score = sum(1 for indicator in real_indicators if indicator in text_lower)
    emotional_score = sum(1 for indicator in emotional_indicators if indicator in text_lower)
    
    # Text length and structure analysis
    word_count = len(text.split())
    sentence_count = text.count('.') + text.count('!') + text.count('?')
    avg_sentence_length = word_count / max(sentence_count, 1)
    
    # All caps analysis
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    
    # Exclamation marks
    exclamation_ratio = text.count('!') / max(len(text), 1)
    
    # Determine prediction based on multiple factors
    base_score = 0
    
    # Fake indicators
    if fake_score > 0:
        base_score += fake_score * 15
    
    # Real indicators
    if real_score > 0:
        base_score -= real_score * 10
    
    # Emotional manipulation
    if emotional_score > 2:
        base_score += emotional_score * 8
    
    # Text structure analysis
    if avg_sentence_length < 10:  # Very short sentences
        base_score += 5
    if caps_ratio > 0.1:  # Too many caps
        base_score += 10
    if exclamation_ratio > 0.01:  # Too many exclamations
        base_score += 8
    
    # Determine final prediction with more decisive thresholds
    if base_score > 10:  # Even lower threshold for FAKE
        prediction = 'FAKE'
        confidence = min(95, 70 + base_score)
    elif base_score < -3:  # Even lower threshold for TRUE
        prediction = 'TRUE'
        confidence = min(95, 70 + abs(base_score))
    else:
        # For neutral cases, be more decisive
        if word_count > 10 and avg_sentence_length > 6:  # Reasonably structured text
            prediction = 'TRUE'
            confidence = 65
        elif caps_ratio > 3 or exclamation_ratio > 0.3:  # Poor formatting
            prediction = 'FAKE'
            confidence = 60
        else:
            # Default to TRUE for neutral content (more user-friendly)
            prediction = 'TRUE'
            confidence = 55
    
    return {
        'prediction': prediction,
        'confidence': confidence,
        'analysis_details': {
            'fake_indicators_found': fake_score,
            'real_indicators_found': real_score,
            'emotional_indicators_found': emotional_score,
            'text_length': word_count,
            'avg_sentence_length': round(avg_sentence_length, 1),
            'caps_ratio': round(caps_ratio * 100, 1),
            'exclamation_ratio': round(exclamation_ratio * 100, 1)
        }
    }

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

# Simple user storage
USERS_FILE = 'commercial/users.json'
USAGE_FILE = 'commercial/usage.json'
FEEDBACK_FILE = 'commercial/feedback.json'

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

def load_feedback():
    """Load feedback from file"""
    try:
        with open(FEEDBACK_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_feedback(feedback):
    """Save feedback to file"""
    os.makedirs('commercial', exist_ok=True)
    with open(FEEDBACK_FILE, 'w') as f:
        json.dump(feedback, f, indent=2)

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

# Routes

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/commercial')
def commercial_home():
    """Commercial home page"""
    return render_template('commercial.html')

@app.route('/pricing')
def pricing():
    """Pricing page showing all subscription plans"""
    return render_template('pricing.html')

@app.route('/payment')
def payment():
    """Payment page"""
    return render_template('payment.html')

@app.route('/commercial/plans')
def get_plans():
    """Get subscription plans"""
    return jsonify({'plans': COMMERCIAL_PLANS})

@app.route('/commercial/register', methods=['GET', 'POST'])
def register():
    """Register new user - GET shows form, POST processes registration"""
    if request.method == 'GET':
        plan = request.args.get('plan', 'starter')
        return render_template('commercial.html', page='register', plan=plan)
    
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

@app.route('/commercial/login', methods=['GET', 'POST'])
def login():
    """User login - GET shows form, POST processes login"""
    if request.method == 'GET':
        return render_template('commercial.html', page='login')
    
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

@app.route('/commercial-info')
def commercial_info():
    """Commercial information page"""
    return jsonify({
        'commercial': True,
        'features': [
            'Subscription-based pricing',
            'Usage tracking',
            'API access',
            'User authentication',
            'Dashboard analytics'
        ],
        'plans': COMMERCIAL_PLANS
    })

@app.route('/submit-feedback', methods=['POST'])
def submit_feedback():
    """Submit user feedback"""
    try:
        # Check if request has JSON data
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
            
        data = request.get_json()
        
        # Debug: Print received data
        print(f"Received feedback data: {data}")
        
        # Validate required fields
        required_fields = ['name', 'email', 'feedback']
        missing_fields = [field for field in required_fields if not data.get(field)]
        
        if missing_fields:
            return jsonify({'error': f'Missing required fields: {", ".join(missing_fields)}'}), 400
        
        # Load existing feedback
        feedback_list = load_feedback()
        
        # Add new feedback
        feedback_entry = {
            'id': len(feedback_list) + 1,
            'name': data['name'].strip(),
            'email': data['email'].strip(),
            'feedback': data['feedback'].strip(),
            'timestamp': datetime.now().isoformat(),
            'status': 'approved'  # Auto-approve for demo
        }
        
        feedback_list.append(feedback_entry)
        save_feedback(feedback_list)
        
        print(f"Feedback saved successfully: {feedback_entry}")
        return jsonify({'message': 'Feedback submitted successfully', 'success': True}), 200
        
    except Exception as e:
        print(f"Error in submit_feedback: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/get-feedback')
def get_feedback():
    """Get approved feedback for display"""
    try:
        feedback_list = load_feedback()
        # Return only approved feedback, limited to 6 most recent
        approved_feedback = [f for f in feedback_list if f.get('status') == 'approved'][-6:]
        return jsonify({'feedback': approved_feedback}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test-feedback', methods=['POST'])
def test_feedback():
    """Test route for feedback debugging"""
    try:
        print("Test feedback route called")
        print("Request method:", request.method)
        print("Request content type:", request.content_type)
        print("Request is_json:", request.is_json)
        
        if request.is_json:
            data = request.get_json()
            print("JSON data received:", data)
        else:
            print("No JSON data received")
            
        return jsonify({'message': 'Test successful', 'received_data': data if request.is_json else None}), 200
    except Exception as e:
        print(f"Test error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/commercial/dashboard')
def dashboard():
    """User dashboard"""
    if 'user_email' not in session:
        return jsonify({'error': 'Authentication required'}), 401
    
    user_email = session['user_email']
    user_plan = session.get('user_plan', 'starter')
    
    # Get user data
    users = load_users()
    user = users.get(user_email, {})
    
    # Get usage data (mock for now)
    usage_data = {
        'analyses_used': 0,
        'analyses_limit': COMMERCIAL_PLANS[user_plan]['analyses_limit'],
        'plan': user_plan,
        'user_name': user.get('name', 'User')
    }
    
    return render_template('commercial.html', page='dashboard', usage=usage_data)

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
        
        # Use enhanced analysis
        try:
            result = enhanced_analysis(text)
            prediction = result['prediction']
            confidence = result['confidence']
            analysis_details = result['analysis_details']
            
            # Create detailed explanation
            explanation_parts = [f"Analysis result: {prediction} (confidence: {confidence}%)"]
            
            if analysis_details['fake_indicators_found'] > 0:
                explanation_parts.append(f"Found {analysis_details['fake_indicators_found']} fake news indicators")
            if analysis_details['real_indicators_found'] > 0:
                explanation_parts.append(f"Found {analysis_details['real_indicators_found']} credible news indicators")
            if analysis_details['emotional_indicators_found'] > 0:
                explanation_parts.append(f"Found {analysis_details['emotional_indicators_found']} emotional manipulation indicators")
            
            if analysis_details['caps_ratio'] > 10:
                explanation_parts.append(f"High use of capital letters ({analysis_details['caps_ratio']}%)")
            if analysis_details['exclamation_ratio'] > 1:
                explanation_parts.append(f"Excessive exclamation marks ({analysis_details['exclamation_ratio']}%)")
            
            explanation = " | ".join(explanation_parts)
            
            individual_results = {
                'enhanced_heuristic': {
                    'model_name': 'Enhanced Heuristic Analysis',
                    'prediction': prediction,
                    'confidence': confidence,
                    'probability_fake': confidence if prediction == 'FAKE' else 100 - confidence,
                    'probability_true': 100 - confidence if prediction == 'FAKE' else confidence,
                    'analysis_details': analysis_details
                }
            }
            
        except Exception as e:
            print(f"Error in enhanced analysis: {e}")
            # Fallback to simple heuristic
            prediction = 'UNKNOWN'
            confidence = 50
            explanation = "Analysis result: UNKNOWN (confidence: 50%) - Error in analysis"
            individual_results = {
                'error': {
                    'model_name': 'Error Fallback',
                    'prediction': prediction,
                    'confidence': confidence,
                    'probability_fake': 50,
                    'probability_true': 50
                }
            }
        
        # Build response
        response = {
            'prediction': prediction,
            'confidence': confidence,
            'news_api_results': {'found': False, 'articles': [], 'error': 'NewsAPI verification requires Professional plan or higher'},
            'individual_results': individual_results,
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
        
        # Extract content from URL and analyze it
        try:
            import requests
            from bs4 import BeautifulSoup
            
            # Fetch URL content
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response_obj = requests.get(url, headers=headers, timeout=10)
            response_obj.raise_for_status()
            
            # Parse HTML content
            soup = BeautifulSoup(response_obj.content, 'html.parser')
            
            # Extract title and main content
            title = soup.find('title')
            title_text = title.get_text().strip() if title else ""
            
            # Try to find main article content
            article_content = ""
            
            # Look for common article content selectors
            content_selectors = [
                'article', '.article-content', '.post-content', '.entry-content',
                '.content', 'main', '.main-content', '[role="main"]'
            ]
            
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    # Remove script and style elements
                    for script in content_elem(["script", "style"]):
                        script.decompose()
                    article_content = content_elem.get_text().strip()
                    break
            
            # If no specific content found, get all text
            if not article_content:
                for script in soup(["script", "style"]):
                    script.decompose()
                article_content = soup.get_text().strip()
            
            # Combine title and content for analysis
            full_text = f"{title_text}\n\n{article_content}"
            
            # Limit text length for analysis
            if len(full_text) > 2000:
                full_text = full_text[:2000] + "..."
            
            # Use enhanced analysis on the extracted content
            result = enhanced_analysis(full_text)
            prediction = result['prediction']
            confidence = result['confidence']
            analysis_details = result['analysis_details']
            
            # Create detailed explanation
            explanation_parts = [f"URL analysis result: {prediction} (confidence: {confidence}%)"]
            explanation_parts.append(f"Analyzed content from: {url}")
            
            if analysis_details['fake_indicators_found'] > 0:
                explanation_parts.append(f"Found {analysis_details['fake_indicators_found']} fake news indicators")
            if analysis_details['real_indicators_found'] > 0:
                explanation_parts.append(f"Found {analysis_details['real_indicators_found']} credible news indicators")
            if analysis_details['emotional_indicators_found'] > 0:
                explanation_parts.append(f"Found {analysis_details['emotional_indicators_found']} emotional manipulation indicators")
            
            if analysis_details['caps_ratio'] > 10:
                explanation_parts.append(f"High use of capital letters ({analysis_details['caps_ratio']}%)")
            if analysis_details['exclamation_ratio'] > 1:
                explanation_parts.append(f"Excessive exclamation marks ({analysis_details['exclamation_ratio']}%)")
            
            explanation = " | ".join(explanation_parts)
            
            individual_results = {
                'enhanced_heuristic': {
                    'model_name': 'Enhanced Heuristic Analysis',
                    'prediction': prediction,
                    'confidence': confidence,
                    'probability_fake': confidence if prediction == 'FAKE' else 100 - confidence,
                    'probability_true': 100 - confidence if prediction == 'FAKE' else confidence,
                    'analysis_details': analysis_details
                }
            }
            
            response = {
                'prediction': prediction,
                'confidence': confidence,
                'news_api_results': {'found': False, 'articles': [], 'error': 'NewsAPI verification requires Professional plan or higher'},
                'individual_results': individual_results,
                'timestamp': datetime.now().isoformat(),
                'url': url,
                'article_title': title_text,
                'article_source': url,
                'article_text': full_text[:200] + '...' if len(full_text) > 200 else full_text,
                'explanation': explanation,
                'commercial_info': {
                    'user_plan': user_plan,
                    'authenticated': True,
                    'feature_restrictions': {
                        'news_api_verification': not plan['restrictions'].get('news_api_verification', False)
                    }
                }
            }
            
        except requests.exceptions.RequestException as e:
            return jsonify({
                'error': f'Failed to fetch URL content: {str(e)}',
                'prediction': 'UNKNOWN',
                'confidence': 0
            }), 400
        except Exception as e:
            return jsonify({
                'error': f'Failed to analyze URL content: {str(e)}',
                'prediction': 'UNKNOWN', 
                'confidence': 0
            }), 400
        
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
        if user_plan == 'starter':
            page_size = min(page_size, 5)
        elif user_plan == 'professional':
            page_size = min(page_size, 10)
        elif user_plan == 'business':
            page_size = min(page_size, 25)
        # Enterprise has no limit
        
        # Mock news articles
        mock_articles = [
            {
                'title': f'Sample News Article {i+1}',
                'description': f'This is a sample news article {i+1} for testing purposes.',
                'url': f'https://example.com/article{i+1}',
                'source': 'Sample News',
                'published_at': '2024-10-13',
                'credibility_score': 0.8,
                'prediction': 'TRUE',
                'confidence': 85
            }
            for i in range(min(page_size, 3))
        ]
        
        # Track usage
        track_usage(user_email)
        
        return jsonify({'articles': mock_articles})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/fetch-news-public', methods=['POST'])
def fetch_news_public():
    """Fetch latest news for public use (no authentication required)"""
    try:
        # Get request data
        data = request.get_json() or {}
        country = data.get('country', 'us')
        category = data.get('category', 'general')
        page_size = min(data.get('page_size', 5), 5)  # Limit to 5 for public use
        
        # Create mock news articles for public use
        mock_articles = [
            {
                'title': f'Sample News Article {i+1} - {category.title()} News',
                'description': f'This is a sample news article about {category} topics. You can analyze this text to test our fake news detection system.',
                'url': f'https://example.com/news/{i+1}',
                'source': 'Sample News',
                'published_at': '2024-10-16',
                'credibility_score': 0.7 + (i * 0.1),
                'prediction': 'TRUE' if i % 2 == 0 else 'FALSE',
                'confidence': 75 + (i * 5)
            }
            for i in range(page_size)
        ]
        
        return jsonify({'articles': mock_articles})
        
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
    print("🚀 Starting Simple Commercial Fake News Detector...")
    print("💰 Subscription Plans Available:")
    print("   📦 Starter: $19/month - 500 analyses")
    print("   💼 Professional: $99/month - 5,000 analyses + URL analysis")
    print("   🏢 Business: $299/month - 25,000 analyses + batch processing")
    print("   🏛️ Enterprise: $999/month - Unlimited analyses")
    
    print("✅ Application ready!")
    
    # Print registered routes
    print("\n📋 Available Routes:")
    print("   🏠 / - Main application")
    print("   💰 /commercial - Commercial features")
    print("   🔍 /analyze - Text analysis (with usage limits)")
    print("   🌐 /analyze-url - URL analysis (Professional+)")
    print("   📰 /fetch-news - News fetching (authenticated)")
    print("   📰 /fetch-news-public - Public news fetching (no auth)")
    print("   📊 /commercial/dashboard - User dashboard")
    print("   💳 /commercial/plans - Subscription plans")
    print("   🔐 /commercial/login - User login")
    print("   📝 /commercial/register - User registration")
    
    # Run the application
    print(f"\n🌐 Starting server on 0.0.0.0:5000")
    print("🔗 Access the application at: http://localhost:5000")
    print("💡 Commercial features are now integrated!")
    
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        use_reloader=False,
        threaded=True
    )
