"""
Integration with existing Flask app
Adds commercial features to the existing fake news detector
"""

from flask import Flask, request, jsonify, session
from datetime import datetime
import sys
import os

# Add commercial modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def integrate_commercial_features(app):
    """
    Integrate commercial features with existing Flask app
    This function modifies the existing app to add subscription features
    """
    
    # Import commercial modules
    from commercial.auth.user_manager import UserManager
    from commercial.api_limits.usage_tracker import UsageTracker
    from commercial.subscriptions.plans import SubscriptionPlans
    from commercial.dashboard.dashboard_manager import DashboardManager
    
    # Initialize managers
    user_manager = UserManager()
    usage_tracker = UsageTracker()
    dashboard_manager = DashboardManager()
    
    # Store managers in app context for access in routes
    app.commercial_managers = {
        'user_manager': user_manager,
        'usage_tracker': usage_tracker,
        'dashboard_manager': dashboard_manager
    }
    
    def require_auth(f):
        """Decorator to require authentication for commercial features"""
        def decorated_function(*args, **kwargs):
            user_email = session.get('user_email')
            if not user_email:
                return jsonify({'error': 'Authentication required. Please login or register.'}), 401
            return f(*args, **kwargs)
        decorated_function.__name__ = f.__name__
        return decorated_function
    
    def check_usage_limit(user_email, user_plan, analysis_type='text'):
        """Check if user has exceeded usage limits"""
        # Get usage limit check
        limit_check = usage_tracker.check_usage_limit(user_email, user_plan)
        
        if not limit_check['allowed']:
            return {
                'allowed': False,
                'message': limit_check['message'],
                'upgrade_required': True
            }
        
        # Track the analysis
        usage_tracker.track_analysis(user_email, analysis_type)
        
        return {'allowed': True}
    
    def check_feature_access(user_plan, feature):
        """Check if user's plan allows access to a feature"""
        return SubscriptionPlans.is_feature_allowed(user_plan, feature)
    
    # Override existing routes with commercial features
    
    @app.route('/analyze', methods=['POST'])
    def analyze_with_limits():
        """Enhanced analyze route with usage limits"""
        try:
            # Check authentication
            user_email = session.get('user_email')
            user_plan = session.get('user_plan', 'starter')
            
            if not user_email:
                # Allow free usage for non-authenticated users (limited)
                user_email = 'anonymous'
                user_plan = 'free'
            
            # Check usage limits
            if user_email != 'anonymous':
                limit_check = check_usage_limit(user_email, user_plan, 'text')
                if not limit_check['allowed']:
                    return jsonify({
                        'error': limit_check['message'],
                        'upgrade_required': True,
                        'current_plan': user_plan,
                        'available_plans': list(SubscriptionPlans.get_all_plans().keys())
                    }), 402  # Payment Required
            
            # Get original data
            data = request.get_json()
            text = data.get('text', '').strip()
            
            if not text:
                return jsonify({'error': 'No text provided'}), 400
            
            # Import existing predictor (from your original app)
            from utils.predictor import UnifiedPredictor
            from utils.model_loader import ModelLoader
            
            # Initialize predictor if not already done
            if not hasattr(app, 'predictor'):
                model_loader = ModelLoader()
                model_loader.load_all_models()
                app.predictor = UnifiedPredictor(model_loader)
            
            # Get ML prediction
            ml_result = app.predictor.ensemble_predict_majority(text)
            
            # Get NewsAPI verification if available and allowed
            news_api_results = {'found': False, 'articles': [], 'error': None}
            if hasattr(app, 'news_verifier') and app.news_verifier:
                if user_email == 'anonymous' or check_feature_access(user_plan, 'news_api_verification'):
                    try:
                        news_api_results = app.news_verifier.verify_news(text)
                    except Exception as e:
                        news_api_results['error'] = str(e)
                else:
                    news_api_results['error'] = 'NewsAPI verification requires Professional plan or higher'
            
            # Generate explanation
            explanation = generate_explanation(ml_result, news_api_results)
            
            # Build response
            response = {
                'prediction': ml_result.get('final_prediction', 'UNKNOWN'),
                'confidence': ml_result.get('confidence', 0),
                'news_api_results': news_api_results,
                'individual_results': ml_result.get('individual_results', {}),
                'timestamp': datetime.now().isoformat(),
                'text': text[:100] + '...' if len(text) > 100 else text,
                'explanation': explanation,
                'commercial_info': {
                    'user_plan': user_plan,
                    'feature_restrictions': {
                        'news_api_verification': not check_feature_access(user_plan, 'news_api_verification'),
                        'url_analysis': not check_feature_access(user_plan, 'url_analysis'),
                        'batch_processing': not check_feature_access(user_plan, 'batch_processing')
                    }
                }
            }
            
            # Log activity for authenticated users
            if user_email != 'anonymous':
                dashboard_manager.log_activity(user_email, 'text_analysis', {
                    'text_length': len(text),
                    'prediction': ml_result.get('final_prediction'),
                    'confidence': ml_result.get('confidence')
                })
            
            return jsonify(response)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/analyze-url', methods=['POST'])
    def analyze_url_with_limits():
        """Enhanced URL analysis with subscription limits"""
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
            if not check_feature_access(user_plan, 'url_analysis'):
                return jsonify({
                    'error': 'URL analysis requires Professional plan or higher',
                    'upgrade_required': True,
                    'current_plan': user_plan,
                    'required_plan': 'professional'
                }), 402
            
            # Check usage limits
            limit_check = check_usage_limit(user_email, user_plan, 'url')
            if not limit_check['allowed']:
                return jsonify({
                    'error': limit_check['message'],
                    'upgrade_required': True
                }), 402
            
            # Get original data
            data = request.get_json()
            url = data.get('url', '').strip()
            
            if not url:
                return jsonify({'error': 'No URL provided'}), 400
            
            # Import existing URL extraction function
            from app import extract_article_content
            
            # Extract content from URL
            try:
                article_content = extract_article_content(url)
                if not article_content:
                    return jsonify({'error': 'Could not extract content from URL'}), 400
            except Exception as e:
                return jsonify({'error': f'Failed to extract content: {str(e)}'}), 400
            
            # Get ML prediction
            if not hasattr(app, 'predictor'):
                from utils.predictor import UnifiedPredictor
                from utils.model_loader import ModelLoader
                model_loader = ModelLoader()
                model_loader.load_all_models()
                app.predictor = UnifiedPredictor(model_loader)
            
            ml_result = app.predictor.ensemble_predict_majority(article_content['text'])
            
            # Get NewsAPI verification if allowed
            news_api_results = {'found': False, 'articles': [], 'error': None}
            if hasattr(app, 'news_verifier') and app.news_verifier:
                if check_feature_access(user_plan, 'news_api_verification'):
                    try:
                        news_api_results = app.news_verifier.verify_news(article_content['text'])
                    except Exception as e:
                        news_api_results['error'] = str(e)
                else:
                    news_api_results['error'] = 'NewsAPI verification requires Professional plan or higher'
            
            # Generate explanation
            explanation = generate_explanation(ml_result, news_api_results)
            
            # Build response
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
                'explanation': explanation,
                'commercial_info': {
                    'user_plan': user_plan,
                    'feature_restrictions': {
                        'news_api_verification': not check_feature_access(user_plan, 'news_api_verification')
                    }
                }
            }
            
            # Log activity
            dashboard_manager.log_activity(user_email, 'url_analysis', {
                'url': url,
                'article_title': article_content.get('title'),
                'prediction': ml_result.get('final_prediction'),
                'confidence': ml_result.get('confidence')
            })
            
            return jsonify(response)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/fetch-news', methods=['POST'])
    def fetch_news_with_limits():
        """Enhanced news fetching with subscription limits"""
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
            limit_check = check_usage_limit(user_email, user_plan, 'news')
            if not limit_check['allowed']:
                return jsonify({
                    'error': limit_check['message'],
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
            
            # Import existing news fetcher
            from news_fetcher import NewsFetcher
            
            if not hasattr(app, 'news_fetcher'):
                app.news_fetcher = NewsFetcher()
            
            if not app.news_fetcher:
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
            
            articles = app.news_fetcher.fetch_and_analyze(
                country=country,
                category=category,
                page_size=page_size
            )
            
            # Log activity
            dashboard_manager.log_activity(user_email, 'news_fetch', {
                'country': country,
                'category': category,
                'page_size': page_size,
                'articles_count': len(articles)
            })
            
            return jsonify({'articles': articles})
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    # Add commercial info to existing routes
    @app.route('/commercial-info', methods=['GET'])
    def get_commercial_info():
        """Get commercial information and available plans"""
        try:
            user_email = session.get('user_email')
            user_plan = session.get('user_plan', 'starter')
            
            if user_email:
                # Get usage stats for authenticated users
                usage_stats = usage_tracker.get_usage_stats(user_email)
                plan_info = SubscriptionPlans.get_plan(user_plan)
            else:
                usage_stats = {'monthly_usage': 0, 'total_analyses': 0}
                plan_info = SubscriptionPlans.get_plan('starter')
            
            return jsonify({
                'authenticated': bool(user_email),
                'user_plan': user_plan,
                'usage_stats': usage_stats,
                'plan_info': plan_info,
                'available_plans': SubscriptionPlans.get_all_plans(),
                'features': {
                    'text_analysis': True,  # Always available
                    'url_analysis': check_feature_access(user_plan, 'url_analysis'),
                    'news_api_verification': check_feature_access(user_plan, 'news_api_verification'),
                    'batch_processing': check_feature_access(user_plan, 'batch_processing'),
                    'api_access': check_feature_access(user_plan, 'api_access')
                }
            }), 200
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    print("✅ Commercial features integrated successfully!")
    print("📊 Available subscription plans:")
    for plan_name, plan_info in SubscriptionPlans.get_all_plans().items():
        print(f"   {plan_info['name']}: ${plan_info['price']}/month - {plan_info['analyses_limit']} analyses")
    
    return app

def generate_explanation(ml_result, news_api_results):
    """Generate explanation (imported from original app)"""
    # This function should be imported from your original app.py
    # For now, we'll create a simple version
    prediction = ml_result.get('final_prediction', 'UNKNOWN')
    confidence = ml_result.get('confidence', 0)
    
    if prediction == 'FAKE':
        return f"❌ ANALYSIS RESULT: High probability of misinformation (confidence: {confidence}%)."
    elif prediction == 'TRUE':
        return f"✅ ANALYSIS RESULT: High probability of credible content (confidence: {confidence}%)."
    else:
        return f"⚠️ ANALYSIS RESULT: Inconclusive (confidence: {confidence}%)."
