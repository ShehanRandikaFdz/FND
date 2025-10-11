"""
Model Loader with Memory Optimization for Streamlit
Loads all three models (SVM, LSTM, BERT) with memory-efficient settings
"""

import os
import pickle
import torch
import numpy as np
import streamlit as st
import gc

# Handle TensorFlow imports with error handling
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.text import Tokenizer
    TENSORFLOW_AVAILABLE = True
except ImportError as e:
    st.warning(f"TensorFlow not available: {e}")
    TENSORFLOW_AVAILABLE = False

# Handle Transformers imports with error handling
try:
    from transformers import DistilBertTokenizer, DistilBertModel
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    st.warning(f"Transformers not available: {e}")
    TRANSFORMERS_AVAILABLE = False

class ModelLoader:
    """Load and manage all three models with memory optimization"""
    
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.models = {}
        self.vectorizers = {}
        self.tokenizers = {}
        self.model_status = {}
        
    def load_svm_model(self):
        """Load SVM model and vectorizer"""
        try:
            svm_path = os.path.join(self.models_dir, "new_svm_model.pkl")
            vectorizer_path = os.path.join(self.models_dir, "new_svm_vectorizer.pkl")
            
            if os.path.exists(svm_path) and os.path.exists(vectorizer_path):
                with open(svm_path, 'rb') as f:
                    self.models['svm'] = pickle.load(f)
                with open(vectorizer_path, 'rb') as f:
                    self.vectorizers['svm'] = pickle.load(f)
                
                self.model_status['svm'] = "loaded"
                return True
            else:
                self.model_status['svm'] = "not_found"
                return False
                
        except Exception as e:
            st.error(f"Error loading SVM model: {e}")
            self.model_status['svm'] = "error"
            return False
    
    def load_lstm_model(self):
        """Load LSTM model and tokenizer"""
        if not TENSORFLOW_AVAILABLE:
            st.warning("TensorFlow not available - skipping LSTM model")
            self.model_status['lstm'] = "tensorflow_unavailable"
            return False
            
        try:
            lstm_path = os.path.join(self.models_dir, "lstm_fake_news_model.h5")
            tokenizer_path = os.path.join(self.models_dir, "lstm_tokenizer.pkl")
            
            if os.path.exists(lstm_path) and os.path.exists(tokenizer_path):
                self.models['lstm'] = load_model(lstm_path)
                with open(tokenizer_path, 'rb') as f:
                    self.tokenizers['lstm'] = pickle.load(f)
                
                self.model_status['lstm'] = "loaded"
                return True
            else:
                self.model_status['lstm'] = "not_found"
                return False
                
        except Exception as e:
            st.error(f"Error loading LSTM model: {e}")
            self.model_status['lstm'] = "error"
            return False
    
    def load_bert_model(self):
        """Load hybrid BERT model (Pre-trained DistilBERT + Logistic Regression) with memory optimization"""
        if not TRANSFORMERS_AVAILABLE:
            st.warning("Transformers not available - skipping BERT model")
            self.model_status['bert'] = "transformers_unavailable"
            return False
            
        try:
            bert_path = os.path.join(self.models_dir, "bert_fake_news_model")
            classifier_path = os.path.join(bert_path, "classifier.pkl")
            
            if os.path.exists(classifier_path):
                # Set memory-efficient settings
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
                
                # Load pre-trained DistilBERT for feature extraction (no local weights needed)
                self.models['bert'] = DistilBertModel.from_pretrained(
                    "distilbert-base-uncased",  # Use pre-trained model
                    torch_dtype=torch.float16,  # Half precision for memory efficiency
                    low_cpu_mem_usage=True
                )
                
                self.tokenizers['bert'] = DistilBertTokenizer.from_pretrained(
                    "distilbert-base-uncased"  # Use pre-trained tokenizer
                )
                
                # Load the logistic regression classifier
                with open(classifier_path, 'rb') as f:
                    self.models['bert_classifier'] = pickle.load(f)
                
                # Set model to evaluation mode and optimize
                self.models['bert'].eval()
                self.models['bert'] = self.models['bert'].half()  # Convert to half precision
                
                self.model_status['bert'] = "loaded"
                return True
            else:
                self.model_status['bert'] = "not_found"
                return False
                
        except Exception as e:
            st.error(f"Error loading BERT model: {e}")
            self.model_status['bert'] = "error"
            return False
    
    def load_all_models(self):
        """Load all available models"""
        st.info("Loading models...")
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Load SVM
        status_text.text("Loading SVM model...")
        svm_loaded = self.load_svm_model()
        progress_bar.progress(33)
        
        # Load LSTM
        status_text.text("Loading LSTM model...")
        lstm_loaded = self.load_lstm_model()
        progress_bar.progress(66)
        
        # Load BERT
        status_text.text("Loading BERT model (this may take a moment)...")
        bert_loaded = self.load_bert_model()
        progress_bar.progress(100)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Display results
        loaded_models = []
        for model_name, status in self.model_status.items():
            if status == "loaded":
                loaded_models.append(model_name.upper())
        
        if loaded_models:
            st.success(f"✅ Successfully loaded models: {', '.join(loaded_models)}")
            return True
        else:
            st.error("❌ No models could be loaded!")
            return False
    
    def get_model_status(self):
        """Return status of all models"""
        return self.model_status
    
    def cleanup_memory(self):
        """Clean up memory after model operations"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

@st.cache_resource
def load_models():
    """Streamlit cached function to load models once"""
    loader = ModelLoader()
    success = loader.load_all_models()
    
    if success:
        return loader
    else:
        return None

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
