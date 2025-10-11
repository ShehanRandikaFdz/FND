"""
Enhanced Credibility Analyzer with Intelligent Ensemble
Combines SVM, LSTM, and BERT models for robust fake news detection.
"""

import numpy as np
import pandas as pd
import re
import pickle
import os
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.linear_model import LogisticRegression
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class CredibilityAnalyzer:
    """
    Advanced credibility analyzer with ensemble methods and uncertainty quantification.
    """
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.models = {}
        self.ensemble_weights = None
        self.confidence_thresholds = {
            'certain_fake': 0.8,
            'certain_real': 0.8,
            'uncertain_lower': 0.45,
            'uncertain_upper': 0.55
        }
        self.load_models()
        self._initialize_ensemble_weights()
    
    def load_models(self):
        """Load all available models for ensemble analysis."""
        print("🔍 Loading Credibility Analyzer models...")
        
        # Load SVM
        try:
            self.models['svm'] = {
                'model': joblib.load(os.path.join(self.models_dir, "new_svm_model.pkl")),
                'vectorizer': joblib.load(os.path.join(self.models_dir, "new_svm_vectorizer.pkl")),
                'accuracy': 0.9959,
                'weight': 0.4,  # Higher weight due to best accuracy
                'type': 'Traditional ML'
            }
            print("✅ SVM model loaded")
        except Exception as e:
            print(f"❌ Error loading SVM model: {e}")
        
        # Load LSTM
        try:
            self.models['lstm'] = {
                'model': tf.keras.models.load_model(f'{self.models_dir}/lstm_fake_news_model.h5'),
                'tokenizer': pickle.load(open(f'{self.models_dir}/lstm_tokenizer.pkl', 'rb')),
                'accuracy': 0.9890,
                'weight': 0.35,  # Good accuracy, sequential understanding
                'type': 'Deep Learning'
            }
            print("✅ LSTM model loaded")
        except Exception as e:
            print(f"❌ Error loading LSTM model: {e}")
        
        # Load BERT
        try:
            self.models['bert'] = {
                'classifier': pickle.load(open(f'{self.models_dir}/bert_fake_news_model/classifier.pkl', 'rb')),
                'tokenizer': AutoTokenizer.from_pretrained(f'{self.models_dir}/bert_fake_news_model'),
                'model': AutoModel.from_pretrained('distilbert-base-uncased'),
                'accuracy': 0.9750,
                'weight': 0.25,  # Lower weight but good contextual understanding
                'type': 'Transformer'
            }
            print("✅ BERT model loaded")
        except Exception as e:
            print(f"❌ Error loading BERT model: {e}")
    
    def _initialize_ensemble_weights(self):
        """Initialize ensemble weights based on model performance."""
        if not self.models:
            return
        
        # Accuracy-based weights (normalized)
        total_accuracy = sum(model['accuracy'] for model in self.models.values())
        self.ensemble_weights = {
            name: model['accuracy'] / total_accuracy 
            for name, model in self.models.items()
        }
        
        print(f"📊 Ensemble weights: {self.ensemble_weights}")
    
    def preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing for credibility analysis."""
        if not isinstance(text, str):
            text = str(text)
        
        # Basic cleaning
        text = text.lower().strip()
        
        # Remove patterns that might indicate fake news
        text = re.sub(r'\[.*?\]', '', text)  # Remove brackets
        text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
        text = re.sub(r'<.*?>+', '', text)  # Remove HTML tags
        text = re.sub(r'\n', ' ', text)  # Replace newlines
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
        
        return text
    
    def extract_meta_features(self, text: str) -> Dict[str, float]:
        """Extract meta-features that indicate credibility."""
        features = {}
        
        # Length features
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())
        
        # Capitalization features
        features['caps_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        
        # Sensational terms (common in fake news)
        sensational_terms = [
            'breaking', 'shocking', 'unbelievable', 'amazing', 'incredible',
            'you won\'t believe', 'click here', 'must see', 'urgent', 'alert'
        ]
        features['sensational_score'] = sum(
            text.lower().count(term) for term in sensational_terms
        ) / max(features['word_count'], 1)
        
        # URL and link indicators
        features['url_count'] = len(re.findall(r'https?://\S+|www\.\S+', text))
        
        return features
    
    def predict_svm(self, text: str) -> Dict[str, float]:
        """Get SVM prediction with confidence."""
        if 'svm' not in self.models:
            return {'prediction': 0.5, 'confidence': 0.0}
        
        cleaned = self.preprocess_text(text)
        vect = self.models['svm']['vectorizer'].transform([cleaned])
        pred = self.models['svm']['model'].predict(vect)[0]
        conf = self.models['svm']['model'].predict_proba(vect)[0].max()
        
        # Convert to probability (1 = real, 0 = fake)
        prob_real = self.models['svm']['model'].predict_proba(vect)[0][1]
        
        return {
            'prediction': float(prob_real),
            'confidence': float(conf),
            'label': 'Real' if pred == 1 else 'Fake'
        }
    
    def predict_lstm(self, text: str) -> Dict[str, float]:
        """Get LSTM prediction with confidence."""
        if 'lstm' not in self.models:
            return {'prediction': 0.5, 'confidence': 0.0}
        
        cleaned = self.preprocess_text(text)
        sequence = self.models['lstm']['tokenizer'].texts_to_sequences([cleaned])
        padded_sequence = pad_sequences(sequence, maxlen=200, padding='post')
        prediction = self.models['lstm']['model'].predict(padded_sequence, verbose=0)[0][0]
        
        # LSTM outputs probability of being real
        prob_real = float(prediction)
        confidence = max(prob_real, 1 - prob_real)
        
        return {
            'prediction': prob_real,
            'confidence': confidence,
            'label': 'Real' if prob_real > 0.5 else 'Fake'
        }
    
    def extract_bert_features(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """Extract BERT features from texts."""
        if 'bert' not in self.models:
            return None
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = self.models['bert']['model']
        tokenizer = self.models['bert']['tokenizer']
        model.to(device)
        model.eval()
        
        features = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            encodings = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors='pt'
            )
            
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                batch_features = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                features.extend(batch_features)
        
        return np.array(features)
    
    def predict_bert(self, text: str) -> Dict[str, float]:
        """Get BERT prediction with confidence."""
        if 'bert' not in self.models:
            return {'prediction': 0.5, 'confidence': 0.0}
        
        cleaned = self.preprocess_text(text)
        features = self.extract_bert_features([cleaned])
        
        prediction_proba = self.models['bert']['classifier'].predict_proba(features)[0]
        predicted_class = self.models['bert']['classifier'].predict(features)[0]
        confidence = prediction_proba.max()
        
        # Convert to probability (1 = real, 0 = fake)
        prob_real = prediction_proba[1] if predicted_class == 1 else prediction_proba[0]
        
        return {
            'prediction': float(prob_real),
            'confidence': float(confidence),
            'label': 'Real' if predicted_class == 1 else 'Fake'
        }
    
    def ensemble_predict(self, text: str) -> Dict[str, any]:
        """
        Intelligent ensemble prediction combining all models.
        """
        # Get individual predictions
        predictions = {}
        if 'svm' in self.models:
            predictions['svm'] = self.predict_svm(text)
        if 'lstm' in self.models:
            predictions['lstm'] = self.predict_lstm(text)
        if 'bert' in self.models:
            predictions['bert'] = self.predict_bert(text)
        
        if not predictions:
            return {'error': 'No models available'}
        
        # Weighted ensemble
        weighted_prediction = 0.0
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for model_name, pred in predictions.items():
            if model_name in self.ensemble_weights:
                weight = self.ensemble_weights[model_name]
                weighted_prediction += pred['prediction'] * weight
                weighted_confidence += pred['confidence'] * weight
                total_weight += weight
        
        if total_weight > 0:
            weighted_prediction /= total_weight
            weighted_confidence /= total_weight
        
        # Determine final label and uncertainty
        if weighted_confidence < self.confidence_thresholds['uncertain_lower']:
            credibility_status = 'uncertain_low_confidence'
        elif (self.confidence_thresholds['uncertain_lower'] <= weighted_prediction <= 
              self.confidence_thresholds['uncertain_upper']):
            credibility_status = 'uncertain_mixed_signals'
        elif weighted_prediction > 0.5 and weighted_confidence >= self.confidence_thresholds['certain_real']:
            credibility_status = 'credible'
        elif weighted_prediction <= 0.5 and weighted_confidence >= self.confidence_thresholds['certain_fake']:
            credibility_status = 'not_credible'
        else:
            credibility_status = 'uncertain'
        
        # Extract meta-features for additional context
        meta_features = self.extract_meta_features(text)
        
        return {
            'credibility_score': float(weighted_prediction),
            'confidence': float(weighted_confidence),
            'credibility_status': credibility_status,
            'label': 'Real' if weighted_prediction > 0.5 else 'Fake',
            'individual_predictions': predictions,
            'meta_features': meta_features,
            'ensemble_weights': self.ensemble_weights
        }
    
    def analyze_credibility(self, text: str) -> Dict[str, any]:
        """
        Main credibility analysis function.
        This is the primary interface for the Credibility Analyzer.
        """
        if not text or not text.strip():
            return {
                'error': 'Empty or invalid text provided',
                'credibility_status': 'invalid_input'
            }
        
        try:
            # Perform ensemble analysis
            result = self.ensemble_predict(text)
            
            # Add analysis metadata
            result['analyzer_version'] = '1.0'
            result['models_used'] = list(self.models.keys())
            result['text_length'] = len(text)
            
            return result
            
        except Exception as e:
            return {
                'error': f'Analysis failed: {str(e)}',
                'credibility_status': 'analysis_error'
            }

def main():
    """Test the enhanced credibility analyzer."""
    print("🔍 Testing Enhanced Credibility Analyzer")
    print("=" * 50)
    
    analyzer = CredibilityAnalyzer()
    
    test_texts = [
        "Reuters reports that the Federal Reserve announced new interest rate policies today after months of economic analysis.",
        "BREAKING: Scientists discover aliens living in your backyard! Click here for shocking photos!",
        "The President signed a new bill into law today, according to official sources.",
        "NASA announces new evidence of water on Mars.",
        "You won't believe what happens next! This one weird trick will change your life!"
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n📰 Test {i}: {text[:60]}...")
        result = analyzer.analyze_credibility(text)
        
        if 'error' not in result:
            print(f"   🎯 Credibility Score: {result['credibility_score']:.4f}")
            print(f"   📊 Confidence: {result['confidence']:.4f}")
            print(f"   🏷️  Status: {result['credibility_status']}")
            print(f"   📝 Label: {result['label']}")
            print(f"   🔍 Meta Features: {result['meta_features']}")
        else:
            print(f"   ❌ Error: {result['error']}")

if __name__ == "__main__":
    main()
