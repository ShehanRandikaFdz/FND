"""
Configuration settings for the Fake News Detector application.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Application configuration class."""
    
    # Flask settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'True') == 'True'
    FLASK_HOST = os.getenv('FLASK_HOST', '0.0.0.0')
    FLASK_PORT = int(os.getenv('FLASK_PORT', '5000'))
    
    # NewsAPI settings
    NEWSAPI_KEY = os.getenv('NEWSAPI_KEY', '')
    
    # Model paths
    MODELS_DIR = 'models'
    SVM_MODEL_PATH = os.path.join(MODELS_DIR, 'new_svm_model.pkl')
    SVM_VECTORIZER_PATH = os.path.join(MODELS_DIR, 'new_svm_vectorizer.pkl')
    LSTM_MODEL_PATH = os.path.join(MODELS_DIR, 'lstm_fake_news_model.h5')
    LSTM_TOKENIZER_PATH = os.path.join(MODELS_DIR, 'lstm_tokenizer.pkl')
    BERT_MODEL_PATH = os.path.join(MODELS_DIR, 'bert_fake_news_model')
    
    # Text processing
    MAX_TEXT_LENGTH = 1000
    MAX_SEQUENCE_LENGTH = 200
    
    # Prediction settings
    CONFIDENCE_THRESHOLD = 0.5
    ENSEMBLE_METHOD = 'majority_voting'