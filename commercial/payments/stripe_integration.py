"""
Stripe Payment Integration
"""

import stripe
import json
from datetime import datetime
from typing import Dict, Optional

class StripePaymentManager:
    """Handle Stripe payment processing"""
    
    def __init__(self, stripe_secret_key: str, stripe_publishable_key: str):
        self.stripe_secret_key = stripe_secret_key
        self.stripe_publishable_key = stripe_publishable_key
        stripe.api_key = stripe_secret_key
        self.payments_file = "commercial/payments/payments.json"
        self.payments = self._load_payments()
    
    def _load_payments(self) -> Dict:
        """Load payment records from file"""
        try:
            with open(self.payments_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def _save_payments(self):
        """Save payment records to file"""
        with open(self.payments_file, 'w') as f:
            json.dump(self.payments, f, indent=2)
    
    def create_customer(self, email: str, name: str) -> Dict:
        """Create Stripe customer"""
        try:
            customer = stripe.Customer.create(
                email=email,
                name=name,
                metadata={
                    'source': 'fake_news_detector'
                }
            )
            
            return {
                'success': True,
                'customer_id': customer.id,
                'message': 'Customer created successfully'
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Error creating customer: {str(e)}'
            }
    
    def create_subscription(self, customer_id: str, plan_name: str) -> Dict:
        """Create subscription for customer"""
        try:
            # Get plan price from our plans configuration
            from ..subscriptions.plans import SubscriptionPlans
            plan = SubscriptionPlans.get_plan(plan_name)
            
            if not plan:
                return {
                    'success': False,
                    'message': 'Invalid plan name'
                }
            
            # Create Stripe price (you would create these in Stripe dashboard)
            price_id = self._get_price_id(plan_name)
            
            subscription = stripe.Subscription.create(
                customer=customer_id,
                items=[{
                    'price': price_id,
                }],
                payment_behavior='default_incomplete',
                payment_settings={'save_default_payment_method': 'on_subscription'},
                expand=['latest_invoice.payment_intent'],
            )
            
            # Record payment
            payment_record = {
                'subscription_id': subscription.id,
                'customer_id': customer_id,
                'plan_name': plan_name,
                'amount': plan['price'],
                'currency': plan['currency'],
                'status': 'pending',
                'created_at': datetime.now().isoformat(),
                'next_billing_date': (datetime.now() + timedelta(days=30)).isoformat()
            }
            
            self.payments[subscription.id] = payment_record
            self._save_payments()
            
            return {
                'success': True,
                'subscription_id': subscription.id,
                'client_secret': subscription.latest_invoice.payment_intent.client_secret,
                'message': 'Subscription created successfully'
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Error creating subscription: {str(e)}'
            }
    
    def cancel_subscription(self, subscription_id: str) -> Dict:
        """Cancel subscription"""
        try:
            subscription = stripe.Subscription.delete(subscription_id)
            
            if subscription_id in self.payments:
                self.payments[subscription_id]['status'] = 'cancelled'
                self.payments[subscription_id]['cancelled_at'] = datetime.now().isoformat()
                self._save_payments()
            
            return {
                'success': True,
                'message': 'Subscription cancelled successfully'
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Error cancelling subscription: {str(e)}'
            }
    
    def get_subscription_status(self, subscription_id: str) -> Dict:
        """Get subscription status"""
        try:
            subscription = stripe.Subscription.retrieve(subscription_id)
            
            return {
                'success': True,
                'status': subscription.status,
                'current_period_start': subscription.current_period_start,
                'current_period_end': subscription.current_period_end,
                'cancel_at_period_end': subscription.cancel_at_period_end
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Error retrieving subscription: {str(e)}'
            }
    
    def _get_price_id(self, plan_name: str) -> str:
        """Get Stripe price ID for plan (configure these in Stripe dashboard)"""
        price_ids = {
            'starter': 'price_starter_monthly',  # Replace with actual Stripe price IDs
            'professional': 'price_professional_monthly',
            'business': 'price_business_monthly',
            'enterprise': 'price_enterprise_monthly'
        }
        return price_ids.get(plan_name, 'price_starter_monthly')
    
    def create_payment_intent(self, amount: int, currency: str = 'usd') -> Dict:
        """Create payment intent for one-time payments"""
        try:
            intent = stripe.PaymentIntent.create(
                amount=amount,
                currency=currency,
                automatic_payment_methods={
                    'enabled': True,
                },
            )
            
            return {
                'success': True,
                'client_secret': intent.client_secret,
                'payment_intent_id': intent.id
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Error creating payment intent: {str(e)}'
            }
    
    def get_payment_history(self, customer_id: str) -> Dict:
        """Get payment history for customer"""
        try:
            payments = stripe.PaymentIntent.list(
                customer=customer_id,
                limit=100
            )
            
            return {
                'success': True,
                'payments': [{
                    'id': payment.id,
                    'amount': payment.amount,
                    'currency': payment.currency,
                    'status': payment.status,
                    'created': payment.created
                } for payment in payments.data]
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Error retrieving payment history: {str(e)}'
            }
