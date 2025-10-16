"""
Commercial API Routes
Integrates with existing Flask app to add subscription features
"""

from flask import Blueprint, request, jsonify, session
from datetime import datetime
import json

# Import commercial modules
from .auth.user_manager import UserManager
from .payments.stripe_integration import StripePaymentManager
from .api_limits.usage_tracker import UsageTracker
from .dashboard.dashboard_manager import DashboardManager
from .subscriptions.plans import SubscriptionPlans

# Create Blueprint
commercial_bp = Blueprint('commercial', __name__, url_prefix='/commercial')

# Initialize managers
user_manager = UserManager()
usage_tracker = UsageTracker()
dashboard_manager = DashboardManager()

# Stripe configuration (set these in your environment)
STRIPE_SECRET_KEY = "sk_test_your_stripe_secret_key"  # Replace with actual key
STRIPE_PUBLISHABLE_KEY = "pk_test_your_stripe_publishable_key"  # Replace with actual key
stripe_manager = StripePaymentManager(STRIPE_SECRET_KEY, STRIPE_PUBLISHABLE_KEY)

# Authentication decorator
def require_auth(f):
    """Decorator to require authentication"""
    def decorated_function(*args, **kwargs):
        user_email = session.get('user_email')
        if not user_email:
            return jsonify({'error': 'Authentication required'}), 401
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

# User Authentication Routes
@commercial_bp.route('/register', methods=['POST'])
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
        
        result = user_manager.register_user(email, password, name, plan)
        
        if result['success']:
            session['user_email'] = email
            session['user_plan'] = plan
            return jsonify(result), 201
        else:
            return jsonify(result), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@commercial_bp.route('/login', methods=['POST'])
def login():
    """User login"""
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        if not all([email, password]):
            return jsonify({'error': 'Email and password required'}), 400
        
        result = user_manager.authenticate_user(email, password)
        
        if result['success']:
            session['user_email'] = email
            session['user_plan'] = result['user']['plan']
            return jsonify(result), 200
        else:
            return jsonify(result), 401
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@commercial_bp.route('/logout', methods=['POST'])
def logout():
    """User logout"""
    session.clear()
    return jsonify({'message': 'Logged out successfully'}), 200

# Subscription Management Routes
@commercial_bp.route('/plans', methods=['GET'])
def get_plans():
    """Get all available subscription plans"""
    try:
        plans = SubscriptionPlans.get_all_plans()
        return jsonify({'success': True, 'plans': plans}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@commercial_bp.route('/subscribe', methods=['POST'])
@require_auth
def subscribe():
    """Create subscription"""
    try:
        data = request.get_json()
        plan_name = data.get('plan')
        
        if not plan_name:
            return jsonify({'error': 'Plan name required'}), 400
        
        user_email = session.get('user_email')
        user_info = user_manager.get_user(user_email)
        
        if not user_info:
            return jsonify({'error': 'User not found'}), 404
        
        # Create Stripe customer if not exists
        customer_result = stripe_manager.create_customer(user_email, user_info['name'])
        if not customer_result['success']:
            return jsonify(customer_result), 400
        
        # Create subscription
        subscription_result = stripe_manager.create_subscription(
            customer_result['customer_id'], 
            plan_name
        )
        
        if subscription_result['success']:
            # Update user plan
            user_manager.update_user_plan(user_email, plan_name)
            session['user_plan'] = plan_name
            
            return jsonify({
                'success': True,
                'subscription_id': subscription_result['subscription_id'],
                'client_secret': subscription_result['client_secret'],
                'message': 'Subscription created successfully'
            }), 200
        else:
            return jsonify(subscription_result), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@commercial_bp.route('/cancel-subscription', methods=['POST'])
@require_auth
def cancel_subscription():
    """Cancel subscription"""
    try:
        data = request.get_json()
        subscription_id = data.get('subscription_id')
        
        if not subscription_id:
            return jsonify({'error': 'Subscription ID required'}), 400
        
        result = stripe_manager.cancel_subscription(subscription_id)
        
        if result['success']:
            # Downgrade user to free plan
            user_email = session.get('user_email')
            user_manager.update_user_plan(user_email, 'starter')
            session['user_plan'] = 'starter'
            
            return jsonify(result), 200
        else:
            return jsonify(result), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Usage Tracking Routes
@commercial_bp.route('/usage', methods=['GET'])
@require_auth
def get_usage():
    """Get user usage statistics"""
    try:
        user_email = session.get('user_email')
        user_plan = session.get('user_plan', 'starter')
        
        # Get usage stats
        usage_stats = usage_tracker.get_usage_stats(user_email)
        
        # Get plan limits
        plan_limit = SubscriptionPlans.get_analyses_limit(user_plan)
        
        return jsonify({
            'success': True,
            'usage_stats': usage_stats,
            'plan_limit': plan_limit,
            'remaining': plan_limit - usage_stats.get('monthly_usage', 0) if plan_limit != -1 else 'unlimited'
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@commercial_bp.route('/check-limit', methods=['GET'])
@require_auth
def check_usage_limit():
    """Check if user has exceeded usage limits"""
    try:
        user_email = session.get('user_email')
        user_plan = session.get('user_plan', 'starter')
        
        result = usage_tracker.check_usage_limit(user_email, user_plan)
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Dashboard Routes
@commercial_bp.route('/dashboard', methods=['GET'])
@require_auth
def get_dashboard():
    """Get user dashboard data"""
    try:
        user_email = session.get('user_email')
        user_plan = session.get('user_plan', 'starter')
        
        dashboard_data = dashboard_manager.get_user_dashboard(user_email, user_plan)
        return jsonify(dashboard_data), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@commercial_bp.route('/feedback', methods=['POST'])
@require_auth
def submit_feedback():
    """Submit user feedback"""
    try:
        data = request.get_json()
        rating = data.get('rating')
        comment = data.get('comment', '')
        
        if not rating or not isinstance(rating, int) or rating < 1 or rating > 5:
            return jsonify({'error': 'Valid rating (1-5) required'}), 400
        
        user_email = session.get('user_email')
        result = dashboard_manager.submit_feedback(user_email, rating, comment)
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Admin Routes
@commercial_bp.route('/admin/analytics', methods=['GET'])
@require_auth
def get_admin_analytics():
    """Get admin analytics (requires admin privileges)"""
    try:
        # In a real implementation, you'd check for admin privileges here
        analytics_summary = dashboard_manager.get_analytics_summary()
        return jsonify(analytics_summary), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@commercial_bp.route('/admin/users', methods=['GET'])
@require_auth
def get_all_users():
    """Get all users (admin function)"""
    try:
        # In a real implementation, you'd check for admin privileges here
        users = user_manager.get_all_users()
        return jsonify({'success': True, 'users': users}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Utility Routes
@commercial_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    }), 200

@commercial_bp.route('/config', methods=['GET'])
def get_config():
    """Get commercial configuration"""
    return jsonify({
        'stripe_publishable_key': STRIPE_PUBLISHABLE_KEY,
        'available_plans': list(SubscriptionPlans.get_all_plans().keys()),
        'features': {
            'authentication': True,
            'subscriptions': True,
            'usage_tracking': True,
            'dashboard': True,
            'payments': True
        }
    }), 200
