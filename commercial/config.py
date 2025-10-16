"""
Commercial Configuration
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class CommercialConfig:
    """Commercial features configuration"""
    
    # Stripe Configuration
    STRIPE_SECRET_KEY = os.getenv('STRIPE_SECRET_KEY', 'sk_test_your_stripe_secret_key')
    STRIPE_PUBLISHABLE_KEY = os.getenv('STRIPE_PUBLISHABLE_KEY', 'pk_test_your_stripe_publishable_key')
    STRIPE_WEBHOOK_SECRET = os.getenv('STRIPE_WEBHOOK_SECRET', 'whsec_your_webhook_secret')
    
    # Database Configuration
    USERS_DB_PATH = 'commercial/auth/users.json'
    PAYMENTS_DB_PATH = 'commercial/payments/payments.json'
    USAGE_DB_PATH = 'commercial/api_limits/usage.json'
    ANALYTICS_DB_PATH = 'commercial/dashboard/analytics.json'
    
    # Security Configuration
    SESSION_TIMEOUT = 24 * 60 * 60  # 24 hours in seconds
    MAX_LOGIN_ATTEMPTS = 5
    LOCKOUT_DURATION = 15 * 60  # 15 minutes in seconds
    
    # Usage Configuration
    FREE_TIER_DAILY_LIMIT = 10  # Free users get 10 analyses per day
    ANONYMOUS_ANALYSES_LIMIT = 3  # Anonymous users get 3 analyses per session
    
    # Feature Flags
    ENABLE_STRIPE = os.getenv('ENABLE_STRIPE', 'True') == 'True'
    ENABLE_ANALYTICS = os.getenv('ENABLE_ANALYTICS', 'True') == 'True'
    ENABLE_USAGE_TRACKING = os.getenv('ENABLE_USAGE_TRACKING', 'True') == 'True'
    
    # Email Configuration (for notifications)
    SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
    SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
    SMTP_USERNAME = os.getenv('SMTP_USERNAME', '')
    SMTP_PASSWORD = os.getenv('SMTP_PASSWORD', '')
    
    # Notification Settings
    SEND_WELCOME_EMAIL = os.getenv('SEND_WELCOME_EMAIL', 'False') == 'True'
    SEND_USAGE_ALERTS = os.getenv('SEND_USAGE_ALERTS', 'True') == 'True'
    USAGE_ALERT_THRESHOLD = 0.8  # Alert when 80% of limit is reached
    
    # Admin Configuration
    ADMIN_EMAIL = os.getenv('ADMIN_EMAIL', 'admin@factcheckpro.com')
    ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD', 'admin123')
    
    # API Configuration
    API_RATE_LIMIT = 100  # requests per minute
    API_RATE_LIMIT_WINDOW = 60  # seconds
    
    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = 'commercial/logs/app.log'
    
    # Backup Configuration
    BACKUP_ENABLED = os.getenv('BACKUP_ENABLED', 'True') == 'True'
    BACKUP_INTERVAL = 24 * 60 * 60  # 24 hours in seconds
    BACKUP_RETENTION_DAYS = 30
    
    @classmethod
    def get_stripe_config(cls):
        """Get Stripe configuration"""
        return {
            'secret_key': cls.STRIPE_SECRET_KEY,
            'publishable_key': cls.STRIPE_PUBLISHABLE_KEY,
            'webhook_secret': cls.STRIPE_WEBHOOK_SECRET
        }
    
    @classmethod
    def get_database_paths(cls):
        """Get database file paths"""
        return {
            'users': cls.USERS_DB_PATH,
            'payments': cls.PAYMENTS_DB_PATH,
            'usage': cls.USAGE_DB_PATH,
            'analytics': cls.ANALYTICS_DB_PATH
        }
    
    @classmethod
    def get_feature_flags(cls):
        """Get feature flags"""
        return {
            'stripe_enabled': cls.ENABLE_STRIPE,
            'analytics_enabled': cls.ENABLE_ANALYTICS,
            'usage_tracking_enabled': cls.ENABLE_USAGE_TRACKING
        }
