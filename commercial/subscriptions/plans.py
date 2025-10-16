"""
Subscription Plans Configuration
"""

class SubscriptionPlans:
    """Define subscription tiers and their features"""
    
    PLANS = {
        'starter': {
            'name': 'Starter',
            'price': 19.00,
            'currency': 'USD',
            'interval': 'month',
            'analyses_limit': 500,
            'features': [
                'Basic text analysis',
                'Email support',
                'Standard confidence scores',
                'Session history (30 days)'
            ],
            'restrictions': {
                'url_analysis': False,
                'news_api_verification': False,
                'batch_processing': False,
                'api_access': False,
                'priority_support': False
            },
            'target_audience': 'Individual journalists, researchers, students'
        },
        
        'professional': {
            'name': 'Professional',
            'price': 99.00,
            'currency': 'USD',
            'interval': 'month',
            'analyses_limit': 5000,
            'features': [
                'All Starter features',
                'URL analysis',
                'NewsAPI verification',
                'API access (5,000 calls/month)',
                'Priority support',
                'Advanced analytics',
                'Session history (90 days)'
            ],
            'restrictions': {
                'url_analysis': True,
                'news_api_verification': True,
                'batch_processing': False,
                'api_access': True,
                'priority_support': True
            },
            'target_audience': 'Small media companies, content creators, PR agencies'
        },
        
        'business': {
            'name': 'Business',
            'price': 299.00,
            'currency': 'USD',
            'interval': 'month',
            'analyses_limit': 25000,
            'features': [
                'All Professional features',
                'Batch processing',
                'Custom integrations',
                'White-label options',
                'Dedicated support',
                'Advanced reporting',
                'Session history (1 year)'
            ],
            'restrictions': {
                'url_analysis': True,
                'news_api_verification': True,
                'batch_processing': True,
                'api_access': True,
                'priority_support': True
            },
            'target_audience': 'News organizations, PR agencies, marketing firms'
        },
        
        'enterprise': {
            'name': 'Enterprise',
            'price': 999.00,
            'currency': 'USD',
            'interval': 'month',
            'analyses_limit': -1,  # Unlimited
            'features': [
                'All Business features',
                'Unlimited analyses',
                'On-premise deployment',
                'Custom model training',
                'SLA guarantees',
                'Dedicated account manager',
                'Unlimited session history'
            ],
            'restrictions': {
                'url_analysis': True,
                'news_api_verification': True,
                'batch_processing': True,
                'api_access': True,
                'priority_support': True
            },
            'target_audience': 'Large media corporations, government agencies, enterprise clients'
        }
    }
    
    @classmethod
    def get_plan(cls, plan_name):
        """Get plan details by name"""
        return cls.PLANS.get(plan_name)
    
    @classmethod
    def get_all_plans(cls):
        """Get all available plans"""
        return cls.PLANS
    
    @classmethod
    def get_plan_features(cls, plan_name):
        """Get features for a specific plan"""
        plan = cls.get_plan(plan_name)
        return plan.get('features', []) if plan else []
    
    @classmethod
    def get_plan_restrictions(cls, plan_name):
        """Get restrictions for a specific plan"""
        plan = cls.get_plan(plan_name)
        return plan.get('restrictions', {}) if plan else {}
    
    @classmethod
    def get_analyses_limit(cls, plan_name):
        """Get analyses limit for a plan (-1 means unlimited)"""
        plan = cls.get_plan(plan_name)
        return plan.get('analyses_limit', 0) if plan else 0
    
    @classmethod
    def is_feature_allowed(cls, plan_name, feature):
        """Check if a feature is allowed for a plan"""
        restrictions = cls.get_plan_restrictions(plan_name)
        return restrictions.get(feature, False)
    
    @classmethod
    def get_plan_price(cls, plan_name):
        """Get price for a plan"""
        plan = cls.get_plan(plan_name)
        return plan.get('price', 0) if plan else 0
