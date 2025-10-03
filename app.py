"""
Enhanced Fake News Detection API with Modern Interactive UI
Complete web interface with article collector, verdict agent, and insights.
"""

from flask import Flask, request, jsonify, render_template_string
import joblib
import os
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.linear_model import LogisticRegression
import numpy as np
import re
import warnings
import datetime
from collections import defaultdict
import json
import unicodedata
import html
import urllib.parse
import time
import logging
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
import math
import psutil
import threading
from collections import deque
import statistics
warnings.filterwarnings('ignore')

# Load models
print("Loading all models...")

# Phase 1 Integration: Enhanced Utilities
class TextPreprocessor:
    """Enhanced text preprocessing with normalization and cleaning."""
    
    def __init__(self):
        self.min_chars = 10
        self.min_words = 3
    
    def preprocess(self, text):
        """Comprehensive text preprocessing."""
        if not text or not isinstance(text, str):
            return "", [], ["Empty or invalid input"]
        
        original_text = text
        errors = []
        
        try:
            # Unicode normalization
            text = unicodedata.normalize('NFKC', text)
            
            # HTML decoding
            text = html.unescape(text)
            
            # Basic cleaning
            text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
            text = re.sub(r'<[^>]+>', '', text)
            
            # URL replacement
            text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
            
            # Whitespace normalization
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Character validation
            if len(text) < self.min_chars:
                errors.append(f"Text too short (minimum {self.min_chars} characters)")
            
            word_count = len(text.split())
            if word_count < self.min_words:
                errors.append(f"Too few words (minimum {self.min_words} words)")
            
            return text, [], errors
            
        except Exception as e:
            return original_text, [], [f"Preprocessing error: {str(e)}"]

class InputValidator:
    """Input validation for security and content quality."""
    
    def __init__(self):
        self.max_length = 10000
        self.max_words = 2000
    
    def validate(self, text):
        """Validate input text."""
        if not text or not isinstance(text, str):
            return False, "Invalid input"
        
        # Length validation
        if len(text) > self.max_length:
            return False, f"Text too long (maximum {self.max_length} characters)"
        
        word_count = len(text.split())
        if word_count > self.max_words:
            return False, f"Too many words (maximum {self.max_words} words)"
        
        # Security checks
        if '<script' in text.lower():
            return False, "Potential security risk detected"
        
        if len(re.findall(r'https?://', text)) > 10:
            return False, "Too many URLs detected"
        
        return True, "Valid input"

class ExplainabilityEngine:
    """Enhanced explainability with detailed reasoning."""
    
    def __init__(self):
        self.fake_patterns = [
            'breaking', 'urgent', 'shocking', 'unbelievable', 'incredible',
            'you won\'t believe', 'doctors hate', 'one weird trick', 'click here',
            'must see', 'must read', 'what happens next', 'this will shock you'
        ]
        
        self.real_patterns = [
            'reuters', 'ap', 'associated press', 'bbc', 'cnn', 'nbc', 'abc', 'cbs',
            'according to', 'study shows', 'research indicates', 'expert says',
            'published in', 'peer-reviewed', 'journal', 'university'
        ]
    
    def analyze_text(self, text, prediction, confidence, model_results):
        """Generate detailed explanation."""
        text_lower = text.lower()
        
        # Extract indicators
        fake_indicators = []
        real_indicators = []
        
        for pattern in self.fake_patterns:
            if pattern in text_lower:
                fake_indicators.append(pattern)
        
        for pattern in self.real_patterns:
            if pattern in text_lower:
                real_indicators.append(pattern)
        
        # Generate insight
        if prediction == 'Real':
            if real_indicators:
                insight = f"This article appears credible because it contains reliable indicators like '{', '.join(real_indicators[:3])}'."
            else:
                insight = f"This article is classified as real with {confidence:.1%} confidence, showing characteristics of legitimate news reporting."
        else:
            if fake_indicators:
                insight = f"This article is likely fake because it contains suspicious indicators like '{', '.join(fake_indicators[:3])}'."
            else:
                insight = f"This article is classified as fake with {confidence:.1%} confidence, likely due to sensational language or lack of credible sources."
        
        # Model agreement analysis
        if model_results:
            labels = [r.get('prediction', 'Unknown') for r in model_results.values() if r]
            unique_labels = set(labels)
            agreement = f"All {len(labels)} models agree" if len(unique_labels) == 1 else f"Models disagree ({len(unique_labels)} different predictions)"
        else:
            agreement = "Single model prediction"
        
        return {
            'fake_indicators': fake_indicators[:8],
            'real_indicators': real_indicators[:8],
            'ai_insight': insight,
            'model_agreement': agreement,
            'confidence_level': self._get_confidence_level(confidence)
        }
    
    def _get_confidence_level(self, confidence):
        """Convert confidence to human-readable level."""
        if confidence >= 0.9:
            return "Very High"
        elif confidence >= 0.8:
            return "High"
        elif confidence >= 0.6:
            return "Moderate"
        elif confidence >= 0.4:
            return "Low"
        else:
            return "Very Low"

class ModelPredictor:
    """Unified predictor for all models"""
    
    def __init__(self):
        self.models = {}
        self.load_models()
        self.analysis_history = []
        
        # Phase 1 Integration: Initialize utilities
        self.preprocessor = TextPreprocessor()
        self.validator = InputValidator()
        self.explainer = ExplainabilityEngine()
    
    def load_models(self):
        """Load all available models"""
        
        # Load SVM
        try:
            self.models['svm'] = {
                'model': joblib.load(os.path.join("models", "final_linear_svm.pkl")),
                'vectorizer': joblib.load(os.path.join("models", "final_vectorizer.pkl")),
                'accuracy': 0.9959,
                'type': 'Traditional ML'
            }
            print("✅ SVM model loaded")
        except Exception as e:
            print(f"❌ Error loading SVM model: {e}")
        
        # Load LSTM
        try:
            self.models['lstm'] = {
                'model': tf.keras.models.load_model('models/lstm_fake_news_model.h5'),
                'tokenizer': pickle.load(open('models/lstm_tokenizer.pkl', 'rb')),
                'accuracy': 0.9890,
                'type': 'Neural Network'
            }
            print("✅ LSTM model loaded")
        except Exception as e:
            print(f"❌ Error loading LSTM model: {e}")
        
        # Load BERT
        try:
            self.models['bert'] = {
                'model': joblib.load('models/bert_fake_news_model/classifier.pkl'),
                'tokenizer': AutoTokenizer.from_pretrained('distilbert-base-uncased'),
                'bert_model': AutoModel.from_pretrained('distilbert-base-uncased'),
                'accuracy': 0.9750,
                'type': 'Transformer'
            }
            print("✅ BERT model loaded")
        except Exception as e:
            print(f"❌ Error loading BERT model: {e}")
    
    def preprocess_text(self, text):
        """Enhanced text preprocessing"""
        text = str(text).lower()
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def predict_svm(self, text):
        """SVM prediction with feature analysis"""
        try:
            cleaned_text = self.preprocess_text(text)
            text_vector = self.models['svm']['vectorizer'].transform([cleaned_text])
            
            # Get prediction and probability
            prediction = self.models['svm']['model'].predict(text_vector)[0]
            probabilities = self.models['svm']['model'].predict_proba(text_vector)[0]
            confidence = max(probabilities)
            
            return {
                'prediction': 'Real' if prediction == 1 else 'Fake',
                'confidence': confidence,
                'model': 'SVM'
            }
        except Exception as e:
            print(f"SVM prediction error: {e}")
            return None
    
    def predict_lstm(self, text):
        """LSTM prediction"""
        try:
            cleaned_text = self.preprocess_text(text)
            tokenizer = self.models['lstm']['tokenizer']
            
            # Tokenize and pad
            sequence = tokenizer.texts_to_sequences([cleaned_text])
            padded_sequence = pad_sequences(sequence, maxlen=1000, padding='post')
            
            # Predict
            prediction = self.models['lstm']['model'].predict(padded_sequence, verbose=0)[0]
            confidence = float(prediction[0])
            
            return {
                'prediction': 'Real' if confidence > 0.5 else 'Fake',
                'confidence': confidence if confidence > 0.5 else 1 - confidence,
                'model': 'LSTM'
            }
        except Exception as e:
            print(f"LSTM prediction error: {e}")
            return None
    
    def predict_bert(self, text):
        """BERT prediction"""
        try:
            cleaned_text = self.preprocess_text(text)
            tokenizer = self.models['bert']['tokenizer']
            
            # Tokenize
            inputs = tokenizer(cleaned_text, return_tensors='pt', truncation=True, padding=True, max_length=512)
            
            # Get BERT embeddings
            with torch.no_grad():
                outputs = self.models['bert']['bert_model'](**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
            
            # Predict with classifier
            prediction = self.models['bert']['model'].predict_proba(embeddings.numpy())[0]
            confidence = max(prediction)
            predicted_class = np.argmax(prediction)
            
            return {
                'prediction': 'Real' if predicted_class == 1 else 'Fake',
                'confidence': confidence,
                'model': 'BERT'
            }
        except Exception as e:
            print(f"BERT prediction error: {e}")
            return None
    
    def analyze_credibility(self, text):
        """Comprehensive credibility analysis with Phase 1 & 2 enhancements"""
        
        # Phase 1: Input validation
        is_valid, validation_message = self.validator.validate(text)
        if not is_valid:
            return {'error': f'Input validation failed: {validation_message}'}
        
        # Phase 1: Enhanced preprocessing
        processed_text, features, preprocessing_errors = self.preprocessor.preprocess(text)
        if preprocessing_errors:
            return {'error': f'Preprocessing failed: {", ".join(preprocessing_errors)}'}
        
        results = {}
        
        # Get predictions from all models (using processed text)
        if 'svm' in self.models:
            results['svm'] = self.predict_svm(processed_text)
        if 'lstm' in self.models:
            results['lstm'] = self.predict_lstm(processed_text)
        if 'bert' in self.models:
            results['bert'] = self.predict_bert(processed_text)
        
        # Calculate ensemble prediction
        valid_results = [r for r in results.values() if r is not None]
        if not valid_results:
            return {'error': 'All models failed to predict'}
        
        # Weighted ensemble (based on model accuracy)
        weights = {'svm': 0.34, 'lstm': 0.33, 'bert': 0.33}
        weighted_sum = 0
        total_weight = 0
        
        for result in valid_results:
            model_name = result['model'].lower()
            weight = weights.get(model_name, 0.33)
            prediction_score = 1 if result['prediction'] == 'Real' else 0
            weighted_sum += prediction_score * weight * result['confidence']
            total_weight += weight * result['confidence']
        
        ensemble_confidence = weighted_sum / total_weight if total_weight > 0 else 0.5
        ensemble_prediction = 'Real' if ensemble_confidence > 0.5 else 'Fake'
        
        # Phase 1: Enhanced explainability
        explanation = self.explainer.analyze_text(text, ensemble_prediction, ensemble_confidence, results)
        
        primary_result = {
            'prediction': ensemble_prediction,
            'confidence': ensemble_confidence,
            'model_results': results,
            'fake_indicators': explanation['fake_indicators'],
            'real_indicators': explanation['real_indicators'],
            'ai_insight': explanation['ai_insight'],
            'model_agreement': explanation['model_agreement'],
            'confidence_level': explanation['confidence_level'],
            'preprocessing_applied': processed_text != text,
            'validation_status': 'valid'
        }
        
        # Store in history
        analysis_record = {
            'text': text[:100] + '...' if len(text) > 100 else text,
            'prediction': primary_result['prediction'],
            'confidence': primary_result['confidence'],
            'timestamp': datetime.datetime.now().isoformat(),
            'model_results': results,
            'validation_status': 'valid',
            'preprocessing_applied': processed_text != text
        }
        self.analysis_history.append(analysis_record)
        
        # Keep only last 50 analyses
        if len(self.analysis_history) > 50:
            self.analysis_history = self.analysis_history[-50:]
        
        return primary_result

# Initialize predictor
predictor = ModelPredictor()

# Flask app
app = Flask(__name__)

# HTML Template (simplified for space)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚨 Fake News Detector</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); min-height: 100vh; padding: 10px; }
        .container { max-width: 100%; margin: 0; padding: 0 20px; }
        .header { text-align: center; color: white; margin-bottom: 20px; }
        .header h1 { font-size: 2rem; margin-bottom: 8px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
        .main-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 25px; margin-bottom: 20px; width: 100%; }
        .card { background: white; border-radius: 12px; padding: 20px; box-shadow: 0 8px 25px rgba(0,0,0,0.1); transition: transform 0.3s ease; }
        .card:hover { transform: translateY(-3px); }
        .input-section { grid-column: 1; }
        .result-section { grid-column: 2; display: none; }
        .result-section.show { display: block; }
        .form-group { margin-bottom: 15px; }
        .form-group label { display: block; margin-bottom: 6px; font-weight: 600; color: #333; font-size: 14px; }
        .form-control { width: 100%; padding: 10px; border: 2px solid #e1e5e9; border-radius: 6px; font-size: 14px; transition: border-color 0.3s ease; }
        .form-control:focus { outline: none; border-color: #ff6b6b; box-shadow: 0 0 0 3px rgba(255, 107, 107, 0.1); }
        textarea.form-control { resize: vertical; min-height: 100px; }
        .btn { background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); color: white; border: none; padding: 12px 25px; border-radius: 6px; font-size: 14px; font-weight: 600; cursor: pointer; transition: all 0.3s ease; width: 100%; }
        .btn:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(255, 107, 107, 0.4); }
        .btn:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
        .verdict-box { text-align: center; padding: 20px; border-radius: 12px; margin-bottom: 20px; position: relative; overflow: hidden; }
        .verdict-box.real { background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%); color: white; }
        .verdict-box.fake { background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%); color: white; }
        .verdict-icon { font-size: 2.5rem; margin-bottom: 10px; animation: pulse 2s infinite; }
        @keyframes pulse { 0% { transform: scale(1); } 50% { transform: scale(1.1); } 100% { transform: scale(1); } }
        .verdict-text { font-size: 1.5rem; font-weight: bold; margin-bottom: 8px; }
        .confidence-score { font-size: 1.2rem; opacity: 0.9; }
        .indicators-section { margin-top: 15px; }
        .indicators-title { font-size: 1rem; font-weight: 600; margin-bottom: 10px; color: #333; }
        .indicators-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
        .indicator-tag { display: inline-block; padding: 4px 8px; margin: 3px; border-radius: 15px; font-size: 11px; font-weight: 500; }
        .fake-tag { background: #ffebee; color: #c62828; border: 1px solid #ffcdd2; }
        .real-tag { background: #e8f5e8; color: #2e7d32; border: 1px solid #c8e6c9; }
        .ai-insight { background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #ff6b6b; margin-top: 15px; }
        .loading { display: none; text-align: center; padding: 20px; }
        .loading.show { display: block; }
        .spinner { border: 4px solid #f3f3f3; border-top: 4px solid #667eea; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto 10px; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        @media (max-width: 768px) { .main-grid { grid-template-columns: 1fr; } .indicators-grid { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚨 Fake News Detector</h1>
            <p>Advanced AI system to detect fake news instantly</p>
        </div>
        
        <div class="main-grid">
            <div class="card input-section">
                <h2 style="margin-bottom: 15px; color: #333; font-size: 18px;">📝 Article Input</h2>
                <form id="analysisForm">
                    <div class="form-group">
                        <label for="articleText">Paste your article here...</label>
                        <textarea id="articleText" class="form-control" placeholder="Paste the article or headline to check its credibility..." required></textarea>
                    </div>
                    <button type="submit" class="btn" id="analyzeBtn">🔍 Detect Fake News</button>
                </form>
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p>Analyzing article credibility...</p>
                </div>
            </div>
            
            <div class="card result-section" id="resultSection">
                <h2 style="color: #333; margin-bottom: 15px; font-size: 18px;">🎯 Detection Result</h2>
                <div class="verdict-box" id="verdictBox">
                    <div class="verdict-icon" id="verdictIcon">⏳</div>
                    <div class="verdict-text" id="verdictText">Analyzing...</div>
                    <div class="confidence-score" id="confidenceScore">Confidence: --%</div>
                </div>
                <div class="indicators-section">
                    <div class="indicators-grid">
                        <div>
                            <div class="indicators-title">🔴 Fake Indicators</div>
                            <div id="fakeIndicators"></div>
                        </div>
                        <div>
                            <div class="indicators-title">🟢 Real Indicators</div>
                            <div id="realIndicators"></div>
                        </div>
                    </div>
                </div>
                <div class="ai-insight">
                    <h4>🧠 AI Insight</h4>
                    <p id="aiInsight">Analysis will appear here...</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('analysisForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            const articleText = document.getElementById('articleText').value.trim();
            if (!articleText) { alert('Please enter some text to analyze.'); return; }
            
            document.getElementById('loading').classList.add('show');
            document.getElementById('analyzeBtn').disabled = true;
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: articleText })
                });
                const result = await response.json();
                if (result.error) throw new Error(result.error);
                displayResults(result);
            } catch (error) {
                alert('Error analyzing article: ' + error.message);
            } finally {
                document.getElementById('loading').classList.remove('show');
                document.getElementById('analyzeBtn').disabled = false;
            }
        });

        function displayResults(result) {
            const resultSection = document.getElementById('resultSection');
            resultSection.classList.add('show');
            
            const verdictBox = document.getElementById('verdictBox');
            const verdictIcon = document.getElementById('verdictIcon');
            const verdictText = document.getElementById('verdictText');
            const confidenceScore = document.getElementById('confidenceScore');
            
            if (result.prediction === 'Real') {
                verdictBox.className = 'verdict-box real';
                verdictIcon.textContent = '✅';
                verdictText.textContent = 'Real News';
            } else {
                verdictBox.className = 'verdict-box fake';
                verdictIcon.textContent = '❌';
                verdictText.textContent = 'Fake News';
            }
            
            confidenceScore.textContent = `Confidence: ${(result.confidence * 100).toFixed(1)}%`;
            
            // Update indicators
            const fakeContainer = document.getElementById('fakeIndicators');
            const realContainer = document.getElementById('realIndicators');
            fakeContainer.innerHTML = '';
            realContainer.innerHTML = '';
            
            (result.fake_indicators || []).forEach(indicator => {
                const tag = document.createElement('span');
                tag.className = 'indicator-tag fake-tag';
                tag.textContent = indicator;
                fakeContainer.appendChild(tag);
            });
            
            (result.real_indicators || []).forEach(indicator => {
                const tag = document.createElement('span');
                tag.className = 'indicator-tag real-tag';
                tag.textContent = indicator;
                realContainer.appendChild(tag);
            });
            
            document.getElementById('aiInsight').textContent = result.ai_insight || 'No insight available.';
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    """Serve the enhanced UI"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze article credibility"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text.strip():
            return jsonify({'error': 'No text provided'}), 400
        
        # Perform analysis
        result = predictor.analyze_credibility(text)
        
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history')
def get_history():
    """Get analysis history"""
    return jsonify({'history': predictor.analysis_history})

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'available_models': list(predictor.models.keys()),
        'total_analyses': len(predictor.analysis_history)
    })

if __name__ == '__main__':
    print("🚀 Starting Enhanced Credibility Analyzer with Modern UI...")
    print("Available endpoints:")
    print("  GET  / - Enhanced UI")
    print("  POST /analyze - Analyze article credibility")
    print("  GET  /history - Get analysis history")
    print("  GET  /health - Health check")
    print("Starting server on http://localhost:5000")
    app.run(debug=False, use_reloader=False, host='0.0.0.0', port=5000)