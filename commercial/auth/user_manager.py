"""
User Management System
"""

import hashlib
import secrets
import json
from datetime import datetime, timedelta
from typing import Dict, Optional, List

class UserManager:
    """Handle user authentication and management"""
    
    def __init__(self, users_file: str = "commercial/auth/users.json"):
        self.users_file = users_file
        self.users = self._load_users()
    
    def _load_users(self) -> Dict:
        """Load users from file"""
        try:
            with open(self.users_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def _save_users(self):
        """Save users to file"""
        with open(self.users_file, 'w') as f:
            json.dump(self.users, f, indent=2)
    
    def _hash_password(self, password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def register_user(self, email: str, password: str, name: str, plan: str = 'starter') -> Dict:
        """Register a new user"""
        if email in self.users:
            return {'success': False, 'message': 'User already exists'}
        
        user_id = secrets.token_hex(16)
        hashed_password = self._hash_password(password)
        
        user_data = {
            'user_id': user_id,
            'email': email,
            'password_hash': hashed_password,
            'name': name,
            'plan': plan,
            'created_at': datetime.now().isoformat(),
            'subscription_start': datetime.now().isoformat(),
            'analyses_used': 0,
            'analyses_reset_date': (datetime.now() + timedelta(days=30)).isoformat(),
            'is_active': True,
            'last_login': None
        }
        
        self.users[email] = user_data
        self._save_users()
        
        return {
            'success': True,
            'message': 'User registered successfully',
            'user_id': user_id
        }
    
    def authenticate_user(self, email: str, password: str) -> Dict:
        """Authenticate user login"""
        if email not in self.users:
            return {'success': False, 'message': 'User not found'}
        
        user = self.users[email]
        hashed_password = self._hash_password(password)
        
        if user['password_hash'] != hashed_password:
            return {'success': False, 'message': 'Invalid password'}
        
        if not user['is_active']:
            return {'success': False, 'message': 'Account is deactivated'}
        
        # Update last login
        user['last_login'] = datetime.now().isoformat()
        self._save_users()
        
        return {
            'success': True,
            'message': 'Login successful',
            'user': {
                'user_id': user['user_id'],
                'email': user['email'],
                'name': user['name'],
                'plan': user['plan'],
                'analyses_used': user['analyses_used'],
                'analyses_limit': self._get_analyses_limit(user['plan'])
            }
        }
    
    def get_user(self, email: str) -> Optional[Dict]:
        """Get user data by email"""
        return self.users.get(email)
    
    def update_user_plan(self, email: str, new_plan: str) -> Dict:
        """Update user subscription plan"""
        if email not in self.users:
            return {'success': False, 'message': 'User not found'}
        
        user = self.users[email]
        user['plan'] = new_plan
        user['subscription_start'] = datetime.now().isoformat()
        user['analyses_used'] = 0  # Reset usage for new plan
        user['analyses_reset_date'] = (datetime.now() + timedelta(days=30)).isoformat()
        
        self._save_users()
        
        return {
            'success': True,
            'message': f'Plan updated to {new_plan}',
            'new_plan': new_plan
        }
    
    def increment_usage(self, email: str) -> Dict:
        """Increment user's analysis usage"""
        if email not in self.users:
            return {'success': False, 'message': 'User not found'}
        
        user = self.users[email]
        user['analyses_used'] += 1
        
        # Check if usage limit exceeded
        limit = self._get_analyses_limit(user['plan'])
        if limit != -1 and user['analyses_used'] > limit:
            return {
                'success': False,
                'message': 'Usage limit exceeded',
                'analyses_used': user['analyses_used'],
                'analyses_limit': limit
            }
        
        self._save_users()
        
        return {
            'success': True,
            'analyses_used': user['analyses_used'],
            'analyses_limit': limit
        }
    
    def reset_usage(self, email: str) -> Dict:
        """Reset user's monthly usage"""
        if email not in self.users:
            return {'success': False, 'message': 'User not found'}
        
        user = self.users[email]
        user['analyses_used'] = 0
        user['analyses_reset_date'] = (datetime.now() + timedelta(days=30)).isoformat()
        
        self._save_users()
        
        return {
            'success': True,
            'message': 'Usage reset successfully'
        }
    
    def _get_analyses_limit(self, plan: str) -> int:
        """Get analyses limit for a plan"""
        from ..subscriptions.plans import SubscriptionPlans
        return SubscriptionPlans.get_analyses_limit(plan)
    
    def get_usage_stats(self, email: str) -> Dict:
        """Get user's usage statistics"""
        if email not in self.users:
            return {'success': False, 'message': 'User not found'}
        
        user = self.users[email]
        limit = self._get_analyses_limit(user['plan'])
        
        return {
            'success': True,
            'analyses_used': user['analyses_used'],
            'analyses_limit': limit,
            'analyses_remaining': limit - user['analyses_used'] if limit != -1 else 'unlimited',
            'plan': user['plan'],
            'reset_date': user['analyses_reset_date']
        }
    
    def deactivate_user(self, email: str) -> Dict:
        """Deactivate user account"""
        if email not in self.users:
            return {'success': False, 'message': 'User not found'}
        
        self.users[email]['is_active'] = False
        self._save_users()
        
        return {
            'success': True,
            'message': 'Account deactivated'
        }
    
    def get_all_users(self) -> List[Dict]:
        """Get all users (admin function)"""
        return list(self.users.values())
