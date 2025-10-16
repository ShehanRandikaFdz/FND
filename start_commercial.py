#!/usr/bin/env python3
"""
Startup Script for Commercial Fake News Detector
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if all required files exist"""
    required_files = [
        'commercial_app.py',
        'commercial/__init__.py',
        'commercial/config.py',
        'commercial/commercial_routes.py',
        'commercial/integration.py',
        'commercial/auth/user_manager.py',
        'commercial/payments/stripe_integration.py',
        'commercial/subscriptions/plans.py',
        'commercial/api_limits/usage_tracker.py',
        'commercial/dashboard/dashboard_manager.py',
        'templates/commercial.html'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        'commercial/auth',
        'commercial/payments', 
        'commercial/subscriptions',
        'commercial/api_limits',
        'commercial/dashboard',
        'commercial/logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def create_sample_env():
    """Create sample .env file if it doesn't exist"""
    env_file = '.env'
    if not os.path.exists(env_file):
        sample_env = """# Commercial Configuration
STRIPE_SECRET_KEY=sk_test_your_stripe_secret_key
STRIPE_PUBLISHABLE_KEY=pk_test_your_stripe_publishable_key
STRIPE_WEBHOOK_SECRET=whsec_your_webhook_secret

# Feature Flags
ENABLE_STRIPE=True
ENABLE_ANALYTICS=True
ENABLE_USAGE_TRACKING=True

# Email Configuration (optional)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password

# Admin Configuration
ADMIN_EMAIL=admin@factcheckpro.com
ADMIN_PASSWORD=admin123
"""
        with open(env_file, 'w') as f:
            f.write(sample_env)
        print(f"✅ Created sample .env file: {env_file}")
        print("⚠️  Please update the .env file with your actual Stripe keys!")

def start_commercial_app():
    """Start the commercial application"""
    print("\n🚀 Starting Commercial Fake News Detector...")
    print("=" * 60)
    
    try:
        # Run the commercial app
        subprocess.run([sys.executable, 'commercial_app.py'], check=True)
    except KeyboardInterrupt:
        print("\n👋 Shutting down Commercial Fake News Detector...")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        return False
    
    return True

def main():
    """Main startup function"""
    print("🎯 Commercial Fake News Detector Startup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('app.py'):
        print("❌ Error: Please run this script from the FND project root directory")
        print("   Current directory:", os.getcwd())
        return False
    
    # Check requirements
    print("\n📋 Checking requirements...")
    if not check_requirements():
        print("\n❌ Requirements check failed!")
        return False
    print("✅ All required files found")
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Create sample .env file
    print("\n⚙️  Setting up configuration...")
    create_sample_env()
    
    # Display startup information
    print("\n🎉 Commercial Fake News Detector is ready!")
    print("=" * 50)
    print("📊 Available Subscription Plans:")
    print("   📦 Starter: $19/month - 500 analyses")
    print("   💼 Professional: $99/month - 5,000 analyses + URL analysis")
    print("   🏢 Business: $299/month - 25,000 analyses + batch processing")
    print("   🏛️ Enterprise: $999/month - Unlimited analyses")
    print("\n🌐 Access URLs:")
    print("   🏠 Main App: http://localhost:5000")
    print("   💰 Commercial: http://localhost:5000/commercial")
    print("   📊 Dashboard: http://localhost:5000/commercial/dashboard")
    print("   🔧 API Config: http://localhost:5000/commercial/config")
    print("\n⚠️  Important Notes:")
    print("   1. Update .env file with your Stripe API keys")
    print("   2. The app will create JSON files for data storage")
    print("   3. For production, consider using a proper database")
    print("   4. Set up SSL certificates for secure payments")
    
    # Ask user if they want to start the app
    print("\n" + "=" * 50)
    response = input("🚀 Start the Commercial Fake News Detector now? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        return start_commercial_app()
    else:
        print("👋 You can start the app later by running: python commercial_app.py")
        return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
