"""
Optimized Model Loader with Lazy Loading and Caching
Reduces initial load time by loading models on-demand
"""

import os
import sys
import time
from typing import Optional, Dict, Any

class OptimizedModelLoader:
    """
    Lazy-loading model manager that loads models only when needed.
    Dramatically reduces initial app startup time.
    """
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self._svm_cache = None
        self._lstm_cache = None
        self._bert_cache = None
        self._credibility_analyzer_cache = None
        
        # Track loading times for performance monitoring
        self.load_times = {}
        
    def get_svm_model(self) -> Dict[str, Any]:
        """Lazy load SVM model - loads only on first call."""
        if self._svm_cache is None:
            start_time = time.time()
            print("⏳ Loading SVM model...")
            
            # Import only when needed
            try:
                from utils.compatibility import safe_load_pickle
            except ImportError:
                import pickle
                def safe_load_pickle(filepath):
                    with open(filepath, 'rb') as f:
                        return pickle.load(f)
            
            try:
                # Try new model paths first
                model = safe_load_pickle(os.path.join(self.models_dir, "new_svm_model.pkl"))
                vectorizer = safe_load_pickle(os.path.join(self.models_dir, "new_svm_vectorizer.pkl"))
            except FileNotFoundError:
                # Fallback to alternative paths
                model = safe_load_pickle(os.path.join(self.models_dir, "final_linear_svm.pkl"))
                vectorizer = safe_load_pickle(os.path.join(self.models_dir, "final_vectorizer.pkl"))
            
            self._svm_cache = {
                'model': model,
                'vectorizer': vectorizer,
                'accuracy': 0.9959
            }
            
            load_time = time.time() - start_time
            self.load_times['svm'] = load_time
            print(f"✅ SVM loaded in {load_time:.2f}s")
            
        return self._svm_cache
    
    def get_lstm_model(self) -> Dict[str, Any]:
        """Lazy load LSTM model - loads only on first call."""
        if self._lstm_cache is None:
            start_time = time.time()
            print("⏳ Loading LSTM model...")
            
            # Import TensorFlow only when needed (saves ~2-3s on startup)
            try:
                from utils.compatibility import safe_load_keras_model, safe_load_pickle
            except ImportError:
                import tensorflow as tf
                import pickle
                def safe_load_keras_model(filepath):
                    return tf.keras.models.load_model(filepath, compile=False)
                def safe_load_pickle(filepath):
                    with open(filepath, 'rb') as f:
                        return pickle.load(f)
            
            try:
                lstm_path = os.path.join(self.models_dir, 'lstm_fake_news_model.h5')
                tokenizer_path = os.path.join(self.models_dir, 'lstm_tokenizer.pkl')
                
                # Load with batch_shape compatibility
                model = safe_load_keras_model(lstm_path)
                tokenizer = safe_load_pickle(tokenizer_path)
                
                self._lstm_cache = {
                    'model': model,
                    'tokenizer': tokenizer,
                    'accuracy': 0.9890
                }
                
                load_time = time.time() - start_time
                self.load_times['lstm'] = load_time
                print(f"✅ LSTM loaded in {load_time:.2f}s")
                
            except FileNotFoundError:
                # Try alternative path
                lstm_path = os.path.join(self.models_dir, 'lstm_best_model.h5')
                model = safe_load_keras_model(lstm_path)
                tokenizer = safe_load_pickle(tokenizer_path)
                
                self._lstm_cache = {
                    'model': model,
                    'tokenizer': tokenizer,
                    'accuracy': 0.9890
                }
                
                load_time = time.time() - start_time
                self.load_times['lstm'] = load_time
                print(f"✅ LSTM loaded in {load_time:.2f}s")
            
        return self._lstm_cache
    
    def get_bert_model(self) -> Dict[str, Any]:
        """Lazy load BERT model - loads only on first call."""
        if self._bert_cache is None:
            start_time = time.time()
            print("⏳ Loading BERT model...")
            
            # Import transformers only when needed (saves ~3-4s on startup)
            try:
                from utils.compatibility import safe_load_transformers_model, safe_load_pickle
            except ImportError:
                from transformers import AutoTokenizer, AutoModel
                import pickle
                
                def safe_load_transformers_model(model_name, local_path=None):
                    model = AutoModel.from_pretrained(
                        local_path if local_path and os.path.exists(local_path) else model_name,
                        low_cpu_mem_usage=False
                    )
                    tokenizer = AutoTokenizer.from_pretrained(
                        local_path if local_path and os.path.exists(local_path) else model_name
                    )
                    return model, tokenizer
                
                def safe_load_pickle(filepath):
                    with open(filepath, 'rb') as f:
                        return pickle.load(f)
            
            try:
                classifier_path = os.path.join(self.models_dir, 'bert_fake_news_model', 'classifier.pkl')
                bert_model_path = os.path.join(self.models_dir, 'bert_fake_news_model')
                
                classifier = safe_load_pickle(classifier_path)
                
                # Try local first, then download from HuggingFace
                try:
                    model, tokenizer = safe_load_transformers_model('distilbert-base-uncased', bert_model_path)
                except:
                    print("📥 Downloading BERT from HuggingFace (first time only)...")
                    model, tokenizer = safe_load_transformers_model('distilbert-base-uncased', None)
                
                self._bert_cache = {
                    'classifier': classifier,
                    'tokenizer': tokenizer,
                    'model': model,
                    'accuracy': 0.9750
                }
                
                load_time = time.time() - start_time
                self.load_times['bert'] = load_time
                print(f"✅ BERT loaded in {load_time:.2f}s")
                
            except Exception as e:
                print(f"⚠️ BERT loading failed: {e}")
                raise
            
        return self._bert_cache
    
    def get_credibility_analyzer(self):
        """Lazy load the full CredibilityAnalyzer with all models."""
        if self._credibility_analyzer_cache is None:
            start_time = time.time()
            print("⏳ Initializing Credibility Analyzer...")
            
            from credibility_analyzer.credibility_analyzer import CredibilityAnalyzer
            
            self._credibility_analyzer_cache = CredibilityAnalyzer(self.models_dir)
            
            load_time = time.time() - start_time
            self.load_times['credibility_analyzer'] = load_time
            print(f"✅ Credibility Analyzer ready in {load_time:.2f}s")
            
        return self._credibility_analyzer_cache
    
    def preload_all_models(self, progress_callback=None):
        """
        Preload all models at once (optional).
        Use this if you want to load everything upfront with progress tracking.
        """
        models = ['svm', 'lstm', 'bert']
        total = len(models)
        
        for idx, model_name in enumerate(models, 1):
            if progress_callback:
                progress_callback(idx / total, f"Loading {model_name.upper()}...")
            
            if model_name == 'svm':
                self.get_svm_model()
            elif model_name == 'lstm':
                self.get_lstm_model()
            elif model_name == 'bert':
                self.get_bert_model()
        
        if progress_callback:
            progress_callback(1.0, "All models loaded!")
        
        return self
    
    def get_load_summary(self) -> str:
        """Get a summary of model loading times."""
        if not self.load_times:
            return "No models loaded yet (lazy loading active)"
        
        total_time = sum(self.load_times.values())
        summary = f"Total loading time: {total_time:.2f}s\n"
        
        for model, load_time in self.load_times.items():
            summary += f"  - {model.upper()}: {load_time:.2f}s\n"
        
        return summary
    
    def is_loaded(self, model_name: str) -> bool:
        """Check if a specific model is already loaded."""
        cache_map = {
            'svm': self._svm_cache,
            'lstm': self._lstm_cache,
            'bert': self._bert_cache,
            'credibility_analyzer': self._credibility_analyzer_cache
        }
        return cache_map.get(model_name) is not None
    
    def clear_cache(self, model_name: Optional[str] = None):
        """Clear model cache to free memory."""
        if model_name:
            if model_name == 'svm':
                self._svm_cache = None
            elif model_name == 'lstm':
                self._lstm_cache = None
            elif model_name == 'bert':
                self._bert_cache = None
            elif model_name == 'credibility_analyzer':
                self._credibility_analyzer_cache = None
            print(f"🗑️ Cleared {model_name} cache")
        else:
            # Clear all caches
            self._svm_cache = None
            self._lstm_cache = None
            self._bert_cache = None
            self._credibility_analyzer_cache = None
            self.load_times.clear()
            print("🗑️ Cleared all model caches")
