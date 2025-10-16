"""
Dashboard Management System
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class DashboardManager:
    """Manage user dashboard and analytics"""
    
    def __init__(self, analytics_file: str = "commercial/dashboard/analytics.json"):
        self.analytics_file = analytics_file
        self.analytics_data = self._load_analytics()
    
    def _load_analytics(self) -> Dict:
        """Load analytics data from file"""
        try:
            with open(self.analytics_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def _save_analytics(self):
        """Save analytics data to file"""
        with open(self.analytics_file, 'w') as f:
            json.dump(self.analytics_data, f, indent=2)
    
    def get_user_dashboard(self, user_email: str, user_plan: str) -> Dict:
        """Get comprehensive dashboard data for user"""
        from ..subscriptions.plans import SubscriptionPlans
        from ..api_limits.usage_tracker import UsageTracker
        from ..auth.user_manager import UserManager
        
        # Initialize managers
        usage_tracker = UsageTracker()
        user_manager = UserManager()
        
        # Get plan information
        plan = SubscriptionPlans.get_plan(user_plan)
        if not plan:
            return {'success': False, 'message': 'Invalid plan'}
        
        # Get usage statistics
        usage_stats = usage_tracker.get_usage_stats(user_email)
        if not usage_stats['success']:
            usage_stats = {
                'total_analyses': 0,
                'monthly_usage': 0,
                'text_analyses': 0,
                'url_analyses': 0,
                'news_fetches': 0,
                'last_analysis': None,
                'reset_date': (datetime.now() + timedelta(days=30)).isoformat()
            }
        
        # Get user information
        user_info = user_manager.get_user(user_email)
        
        # Calculate usage percentage
        limit = plan['analyses_limit']
        usage_percentage = 0
        if limit != -1 and limit > 0:
            usage_percentage = (usage_stats['monthly_usage'] / limit) * 100
        
        # Get recent activity (last 7 days)
        recent_activity = self._get_recent_activity(user_email, days=7)
        
        # Get plan features and restrictions
        features = plan['features']
        restrictions = plan['restrictions']
        
        return {
            'success': True,
            'user_info': {
                'email': user_email,
                'name': user_info.get('name', '') if user_info else '',
                'plan': user_plan,
                'subscription_start': user_info.get('subscription_start', '') if user_info else '',
                'is_active': user_info.get('is_active', True) if user_info else True
            },
            'usage_stats': {
                'total_analyses': usage_stats['total_analyses'],
                'monthly_usage': usage_stats['monthly_usage'],
                'text_analyses': usage_stats['text_analyses'],
                'url_analyses': usage_stats['url_analyses'],
                'news_fetches': usage_stats['news_fetches'],
                'usage_percentage': round(usage_percentage, 1),
                'remaining': limit - usage_stats['monthly_usage'] if limit != -1 else 'unlimited',
                'reset_date': usage_stats['reset_date']
            },
            'plan_info': {
                'name': plan['name'],
                'price': plan['price'],
                'currency': plan['currency'],
                'interval': plan['interval'],
                'analyses_limit': limit,
                'features': features,
                'restrictions': restrictions
            },
            'recent_activity': recent_activity,
            'dashboard_metrics': self._calculate_dashboard_metrics(user_email)
        }
    
    def _get_recent_activity(self, user_email: str, days: int = 7) -> List[Dict]:
        """Get recent activity for user"""
        if user_email not in self.analytics_data:
            return []
        
        user_analytics = self.analytics_data[user_email]
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_activities = []
        for activity in user_analytics.get('activities', []):
            activity_date = datetime.fromisoformat(activity['timestamp'])
            if activity_date >= cutoff_date:
                recent_activities.append(activity)
        
        return sorted(recent_activities, key=lambda x: x['timestamp'], reverse=True)[:10]
    
    def _calculate_dashboard_metrics(self, user_email: str) -> Dict:
        """Calculate dashboard metrics"""
        if user_email not in self.analytics_data:
            return {
                'accuracy_score': 0,
                'reliability_score': 0,
                'efficiency_score': 0,
                'total_savings': 0
            }
        
        user_analytics = self.analytics_data[user_email]
        
        # Calculate accuracy score based on user feedback
        feedback_count = len(user_analytics.get('feedback', []))
        positive_feedback = sum(1 for f in user_analytics.get('feedback', []) if f.get('rating', 0) >= 4)
        accuracy_score = (positive_feedback / feedback_count * 100) if feedback_count > 0 else 0
        
        # Calculate reliability score based on successful analyses
        total_analyses = user_analytics.get('total_analyses', 0)
        successful_analyses = user_analytics.get('successful_analyses', 0)
        reliability_score = (successful_analyses / total_analyses * 100) if total_analyses > 0 else 0
        
        # Calculate efficiency score based on response times
        avg_response_time = user_analytics.get('avg_response_time', 0)
        efficiency_score = max(0, 100 - (avg_response_time / 10))  # Normalize to 0-100
        
        # Calculate total savings (estimated time saved)
        total_savings = total_analyses * 0.5  # Assume 30 seconds saved per analysis
        
        return {
            'accuracy_score': round(accuracy_score, 1),
            'reliability_score': round(reliability_score, 1),
            'efficiency_score': round(efficiency_score, 1),
            'total_savings': round(total_savings, 1)
        }
    
    def log_activity(self, user_email: str, activity_type: str, details: Dict) -> Dict:
        """Log user activity"""
        if user_email not in self.analytics_data:
            self.analytics_data[user_email] = {
                'activities': [],
                'feedback': [],
                'total_analyses': 0,
                'successful_analyses': 0,
                'avg_response_time': 0,
                'created_at': datetime.now().isoformat()
            }
        
        activity = {
            'type': activity_type,
            'timestamp': datetime.now().isoformat(),
            'details': details
        }
        
        self.analytics_data[user_email]['activities'].append(activity)
        self.analytics_data[user_email]['total_analyses'] += 1
        
        # Keep only last 100 activities
        if len(self.analytics_data[user_email]['activities']) > 100:
            self.analytics_data[user_email]['activities'] = self.analytics_data[user_email]['activities'][-100:]
        
        self._save_analytics()
        
        return {'success': True, 'message': 'Activity logged successfully'}
    
    def submit_feedback(self, user_email: str, rating: int, comment: str = '') -> Dict:
        """Submit user feedback"""
        if user_email not in self.analytics_data:
            self.analytics_data[user_email] = {
                'activities': [],
                'feedback': [],
                'total_analyses': 0,
                'successful_analyses': 0,
                'avg_response_time': 0,
                'created_at': datetime.now().isoformat()
            }
        
        feedback = {
            'rating': rating,
            'comment': comment,
            'timestamp': datetime.now().isoformat()
        }
        
        self.analytics_data[user_email]['feedback'].append(feedback)
        self._save_analytics()
        
        return {'success': True, 'message': 'Feedback submitted successfully'}
    
    def get_analytics_summary(self) -> Dict:
        """Get analytics summary for admin dashboard"""
        total_users = len(self.analytics_data)
        total_analyses = sum(user_data.get('total_analyses', 0) for user_data in self.analytics_data.values())
        
        # Calculate average ratings
        all_feedback = []
        for user_data in self.analytics_data.values():
            all_feedback.extend(user_data.get('feedback', []))
        
        avg_rating = 0
        if all_feedback:
            avg_rating = sum(f.get('rating', 0) for f in all_feedback) / len(all_feedback)
        
        return {
            'success': True,
            'total_users': total_users,
            'total_analyses': total_analyses,
            'average_rating': round(avg_rating, 1),
            'total_feedback': len(all_feedback)
        }
