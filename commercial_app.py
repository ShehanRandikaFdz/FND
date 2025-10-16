"""
Commercial Fake News Detector Application
Integrates commercial features with existing Flask app
"""

import os
import sys
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

# Import your existing app configuration
from config import Config

# Initialize Flask app
app = Flask(__name__)
app.secret_key = Config.SECRET_KEY
CORS(app)

# Import and integrate commercial features
from commercial.integration import integrate_commercial_features
from commercial.commercial_routes import commercial_bp

# Register commercial blueprint
app.register_blueprint(commercial_bp)

# Initialize commercial features
app = integrate_commercial_features(app)

# Global variables for ML components (from your original app)
model_loader = None
predictor = None
news_verifier = None
news_fetcher = None

def initialize_ml_components():
    """Initialize ML models and components (from your original app)"""
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

# Routes from your original app (enhanced with commercial features)

@app.route('/')
def index():
    """Serve the main HTML page with commercial features"""
    return render_template('index.html')

@app.route('/history', methods=['GET'])
def get_history():
    """Get analysis history (enhanced with commercial features)"""
    user_email = session.get('user_email')
    
    if user_email:
        # For authenticated users, get history from dashboard
        from commercial.dashboard.dashboard_manager import DashboardManager
        dashboard_manager = DashboardManager()
        dashboard_data = dashboard_manager.get_user_dashboard(user_email, session.get('user_plan', 'starter'))
        
        # Extract recent activities as history
        history = dashboard_data.get('recent_activity', [])
    else:
        # For anonymous users, use session history
        history = session.get('history', [])
    
    return jsonify({'history': history})

@app.route('/clear-history', methods=['POST'])
def clear_history():
    """Clear analysis history"""
    user_email = session.get('user_email')
    
    if user_email:
        # For authenticated users, clear dashboard history
        from commercial.dashboard.dashboard_manager import DashboardManager
        dashboard_manager = DashboardManager()
        # Clear activities (implement this method in dashboard_manager)
        pass
    
    # Clear session history
    session.clear()
    return jsonify({'message': 'History cleared'})

# Commercial-specific routes

@app.route('/pricing', methods=['GET'])
def pricing():
    """Pricing page"""
    from commercial.subscriptions.plans import SubscriptionPlans
    plans = SubscriptionPlans.get_all_plans()
    return jsonify({'plans': plans})

@app.route('/features', methods=['GET'])
def features():
    """Features comparison page"""
    from commercial.subscriptions.plans import SubscriptionPlans
    plans = SubscriptionPlans.get_all_plans()
    
    features_comparison = {}
    for plan_name, plan_info in plans.items():
        features_comparison[plan_name] = {
            'name': plan_info['name'],
            'price': plan_info['price'],
            'features': plan_info['features'],
            'restrictions': plan_info['restrictions']
        }
    
    return jsonify({'features': features_comparison})

@app.route('/user-profile', methods=['GET'])
def user_profile():
    """Get user profile information"""
    user_email = session.get('user_email')
    
    if not user_email:
        return jsonify({'error': 'Not authenticated'}), 401
    
    from commercial.auth.user_manager import UserManager
    from commercial.dashboard.dashboard_manager import DashboardManager
    
    user_manager = UserManager()
    dashboard_manager = DashboardManager()
    
    user_info = user_manager.get_user(user_email)
    user_plan = session.get('user_plan', 'starter')
    dashboard_data = dashboard_manager.get_user_dashboard(user_email, user_plan)
    
    return jsonify({
        'user_info': user_info,
        'dashboard_data': dashboard_data
    })

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
