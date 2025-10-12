"""
Configuration Management for Real-Time News Monitoring
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration"""
    
    # API Keys
    NEWSAPI_KEY = os.getenv('NEWSAPI_KEY', '')
    
    # News Fetching Settings
    DEFAULT_COUNTRY = os.getenv('NEWS_COUNTRY', 'us')  # us, gb, ca, au, etc.
    DEFAULT_LANGUAGE = os.getenv('NEWS_LANGUAGE', 'en')  # en, es, fr, de, etc.
    DEFAULT_PAGE_SIZE = int(os.getenv('NEWS_PAGE_SIZE', '20'))  # Articles per fetch
    
    # News Categories (comma-separated)
    NEWS_CATEGORIES = os.getenv('NEWS_CATEGORIES', 'general,technology,business').split(',')
    
    # Credibility Thresholds
    FAKE_THRESHOLD = float(os.getenv('FAKE_THRESHOLD', '0.4'))  # Below = likely fake
    SUSPICIOUS_THRESHOLD = float(os.getenv('SUSPICIOUS_THRESHOLD', '0.6'))  # Between = suspicious
    CREDIBLE_THRESHOLD = float(os.getenv('CREDIBLE_THRESHOLD', '0.6'))  # Above = credible
    
    # Display Settings
    MAX_ARTICLES_DISPLAY = int(os.getenv('MAX_ARTICLES_DISPLAY', '50'))
    AUTO_REFRESH_SECONDS = int(os.getenv('AUTO_REFRESH_SECONDS', '300'))  # 5 minutes
    
    @classmethod
    def validate(cls) -> bool:
        """Validate configuration"""
        if not cls.NEWSAPI_KEY:
            print("❌ NEWSAPI_KEY not found!")
            print("   Please set it in .env file or environment variable")
            return False
        return True
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("\n" + "="*60)
        print("📋 CURRENT CONFIGURATION")
        print("="*60)
        print(f"API Key: {'✅ Set' if cls.NEWSAPI_KEY else '❌ Not set'}")
        print(f"Country: {cls.DEFAULT_COUNTRY}")
        print(f"Language: {cls.DEFAULT_LANGUAGE}")
        print(f"Page Size: {cls.DEFAULT_PAGE_SIZE}")
        print(f"Categories: {', '.join(cls.NEWS_CATEGORIES)}")
        print(f"Fake Threshold: {cls.FAKE_THRESHOLD}")
        print(f"Suspicious Threshold: {cls.SUSPICIOUS_THRESHOLD}")
        print(f"Credible Threshold: {cls.CREDIBLE_THRESHOLD}")
        print("="*60 + "\n")


# Create singleton instance
config = Config()
