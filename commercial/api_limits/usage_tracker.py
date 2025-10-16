"""
Usage Tracking and API Limits
"""

import json
from datetime import datetime, timedelta
from typing import Dict, Optional

class UsageTracker:
    """Track API usage and enforce limits"""
    
    def __init__(self, usage_file: str = "commercial/api_limits/usage.json"):
        self.usage_file = usage_file
        self.usage_data = self._load_usage_data()
    
    def _load_usage_data(self) -> Dict:
        """Load usage data from file"""
        try:
            with open(self.usage_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def _save_usage_data(self):
        """Save usage data to file"""
        with open(self.usage_file, 'w') as f:
            json.dump(self.usage_data, f, indent=2)
    
    def track_analysis(self, user_email: str, analysis_type: str = 'text') -> Dict:
        """Track a single analysis"""
        if user_email not in self.usage_data:
            self.usage_data[user_email] = {
                'total_analyses': 0,
                'text_analyses': 0,
                'url_analyses': 0,
                'news_fetches': 0,
                'last_analysis': None,
                'monthly_usage': 0,
                'reset_date': (datetime.now() + timedelta(days=30)).isoformat()
            }
        
        user_data = self.usage_data[user_email]
        user_data['total_analyses'] += 1
        user_data['monthly_usage'] += 1
        user_data['last_analysis'] = datetime.now().isoformat()
        
        # Track specific analysis types
        if analysis_type == 'text':
            user_data['text_analyses'] += 1
        elif analysis_type == 'url':
            user_data['url_analyses'] += 1
        elif analysis_type == 'news':
            user_data['news_fetches'] += 1
        
        self._save_usage_data()
        
        return {
            'success': True,
            'total_analyses': user_data['total_analyses'],
            'monthly_usage': user_data['monthly_usage']
        }
    
    def check_usage_limit(self, user_email: str, user_plan: str) -> Dict:
        """Check if user has exceeded usage limits"""
        from ..subscriptions.plans import SubscriptionPlans
        
        plan_limit = SubscriptionPlans.get_analyses_limit(user_plan)
        
        if user_email not in self.usage_data:
            return {
                'allowed': True,
                'usage': 0,
                'limit': plan_limit,
                'remaining': plan_limit if plan_limit != -1 else 'unlimited'
            }
        
        user_data = self.usage_data[user_email]
        current_usage = user_data['monthly_usage']
        
        # Check if limit exceeded
        if plan_limit != -1 and current_usage >= plan_limit:
            return {
                'allowed': False,
                'usage': current_usage,
                'limit': plan_limit,
                'remaining': 0,
                'message': 'Usage limit exceeded. Please upgrade your plan.'
            }
        
        return {
            'allowed': True,
            'usage': current_usage,
            'limit': plan_limit,
            'remaining': plan_limit - current_usage if plan_limit != -1 else 'unlimited'
        }
    
    def reset_monthly_usage(self, user_email: str) -> Dict:
        """Reset monthly usage for user"""
        if user_email not in self.usage_data:
            return {'success': False, 'message': 'User not found'}
        
        self.usage_data[user_email]['monthly_usage'] = 0
        self.usage_data[user_email]['reset_date'] = (datetime.now() + timedelta(days=30)).isoformat()
        
        self._save_usage_data()
        
        return {
            'success': True,
            'message': 'Monthly usage reset successfully'
        }
    
    def get_usage_stats(self, user_email: str) -> Dict:
        """Get comprehensive usage statistics"""
        if user_email not in self.usage_data:
            return {
                'success': False,
                'message': 'No usage data found'
            }
        
        user_data = self.usage_data[user_email]
        
        return {
            'success': True,
            'total_analyses': user_data['total_analyses'],
            'monthly_usage': user_data['monthly_usage'],
            'text_analyses': user_data['text_analyses'],
            'url_analyses': user_data['url_analyses'],
            'news_fetches': user_data['news_fetches'],
            'last_analysis': user_data['last_analysis'],
            'reset_date': user_data['reset_date']
        }
    
    def get_all_usage_stats(self) -> Dict:
        """Get usage statistics for all users (admin function)"""
        return {
            'success': True,
            'total_users': len(self.usage_data),
            'usage_data': self.usage_data
        }
    
    def cleanup_old_data(self, days: int = 90) -> Dict:
        """Clean up old usage data"""
        cutoff_date = datetime.now() - timedelta(days=days)
        cleaned_count = 0
        
        for user_email, user_data in list(self.usage_data.items()):
            if user_data.get('last_analysis'):
                last_analysis = datetime.fromisoformat(user_data['last_analysis'])
                if last_analysis < cutoff_date:
                    del self.usage_data[user_email]
                    cleaned_count += 1
        
        self._save_usage_data()
        
        return {
            'success': True,
            'cleaned_users': cleaned_count,
            'remaining_users': len(self.usage_data)
        }
