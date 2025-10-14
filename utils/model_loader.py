"""
Model Loader for Flask Application
Loads all three models (SVM, LSTM, BERT) with error handling
"""

import os
import pickle
import numpy as np
import gc
from config import Config

# Handle TensorFlow imports with error handling
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.text import Tokenizer
    TENSORFLOW_AVAILABLE = True
except ImportError as e:
    print(f"TensorFlow not available: {e}")
    TENSORFLOW_AVAILABLE = False

# Handle Transformers imports with error handling
try:
    from transformers import DistilBertTokenizer, DistilBertModel
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"Transformers not available: {e}")
    TRANSFORMERS_AVAILABLE = False
except AttributeError as e:
    if "register_pytree_node" in str(e):
        print(f"PyTorch compatibility issue: {e}")
    TRANSFORMERS_AVAILABLE = False

class ModelLoader:
    """Load and manage all three models"""
    
    def __init__(self):
        self.models_dir = Config.MODELS_DIR
        self.models = {}
        self.vectorizers = {}
        self.tokenizers = {}
        self.model_status = {}
        
    def load_svm_model(self):
        """Load SVM model and vectorizer"""
        try:
            if os.path.exists(Config.SVM_MODEL_PATH) and os.path.exists(Config.SVM_VECTORIZER_PATH):
                with open(Config.SVM_MODEL_PATH, 'rb') as f:
                    self.models['svm'] = pickle.load(f)
                with open(Config.SVM_VECTORIZER_PATH, 'rb') as f:
                    self.vectorizers['svm'] = pickle.load(f)
                
                self.model_status['svm'] = "loaded"
                print("SVM model loaded successfully")
                return True
            else:
                self.model_status['svm'] = "not_found"
                print("SVM model files not found")
                return False
                
        except Exception as e:
            print(f"Error loading SVM model: {e}")
            self.model_status['svm'] = "error"
            return False
    
    def load_lstm_model(self):
        """Load LSTM model and tokenizer"""
        if not TENSORFLOW_AVAILABLE:
            print("WARNING: TensorFlow not available - skipping LSTM model")
            self.model_status['lstm'] = "tensorflow_unavailable"
            return False
            
        try:
            if os.path.exists(Config.LSTM_MODEL_PATH) and os.path.exists(Config.LSTM_TOKENIZER_PATH):
                try:
                    # Load with compatibility settings
                    self.models['lstm'] = load_model(Config.LSTM_MODEL_PATH, compile=False)
                    with open(Config.LSTM_TOKENIZER_PATH, 'rb') as f:
                        self.tokenizers['lstm'] = pickle.load(f)
                    
                    self.model_status['lstm'] = "loaded"
                    print(" LSTM model loaded successfully")
                    return True
                except Exception as model_error:
                    print(f"WARNING: LSTM model compatibility issue: {model_error}")
                    self.model_status['lstm'] = "compatibility_error"
                    return False
            else:
                self.model_status['lstm'] = "not_found"
                print(" LSTM model files not found")
                return False
                
        except Exception as e:
            print(f" Error loading LSTM model: {e}")
            self.model_status['lstm'] = "error"
            return False
    
    def load_bert_model(self):
        """Load hybrid BERT model (Pre-trained DistilBERT + Logistic Regression)"""
        if not TRANSFORMERS_AVAILABLE:
            print("WARNING: Transformers not available - skipping BERT model")
            self.model_status['bert'] = "transformers_unavailable"
            return False
            
        try:
            classifier_path = os.path.join(Config.BERT_MODEL_PATH, "classifier.pkl")
            
            if os.path.exists(classifier_path):
                # Load pre-trained DistilBERT for feature extraction
                try:
                    self.models['bert'] = DistilBertModel.from_pretrained(
                        "distilbert-base-uncased",
                        low_cpu_mem_usage=False
                    )
                except Exception as bert_error:
                    error_msg = str(bert_error)
                    if "numpy._core" in error_msg:
                        print("WARNING: BERT model compatibility issue, trying alternative loading...")
                        self.models['bert'] = DistilBertModel.from_pretrained(
                            "distilbert-base-uncased"
                        )
                    else:
                        raise bert_error
                
                self.tokenizers['bert'] = DistilBertTokenizer.from_pretrained(
                    "distilbert-base-uncased"
                )
                
                # Load the logistic regression classifier
                with open(classifier_path, 'rb') as f:
                    self.models['bert_classifier'] = pickle.load(f)
                
                # Set model to evaluation mode
                self.models['bert'].eval()
                
                self.model_status['bert'] = "loaded"
                print(" BERT model loaded successfully")
                return True
            else:
                self.model_status['bert'] = "not_found"
                print(" BERT classifier file not found")
                return False
                
        except Exception as e:
            print(f" Error loading BERT model: {e}")
            self.model_status['bert'] = "error"
            return False
    
    def load_all_models(self):
        """Load all available models"""
        print(" Loading ML models...")
        
        # Load all models
        svm_loaded = self.load_svm_model()
        lstm_loaded = self.load_lstm_model()
        bert_loaded = self.load_bert_model()
        
        # Display results
        loaded_models = []
        for model_name, status in self.model_status.items():
            if status == "loaded":
                loaded_models.append(model_name.upper())
        
        if loaded_models:
            print(f" Successfully loaded models: {', '.join(loaded_models)}")
            return True
        else:
            print(" No models could be loaded!")
            return False
    
    def get_model_status(self):
        """Return status of all models"""
        return self.model_status
    
    def cleanup_memory(self):
        """Clean up memory after model operations"""
        gc.collect()

def get_model_info():
    """Get information about model performance"""
    return {
        'svm': {
            'name': 'Support Vector Machine',
            'accuracy': 99.5,
            'description': 'Traditional ML model with TF-IDF features',
            'strength': 'High accuracy on structured text features'
        },
        'lstm': {
            'name': 'Long Short-Term Memory',
            'accuracy': 87.0,
            'description': 'Deep learning model for sequential data',
            'strength': 'Good at capturing temporal patterns'
        },
        'bert': {
            'name': 'DistilBERT (Hybrid)',
            'accuracy': 89.0,
            'description': 'Pre-trained DistilBERT + Custom Logistic Regression classifier',
            'strength': 'Excellent at understanding context with efficient hybrid approach'
        }
    }