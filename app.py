"""
Enhanced Fake News Detection System
Combines Flask API and Streamlit UI with SVM, LSTM, and BERT Models
Complete implementation with comprehensive analysis and neural network diversity.
"""

import os
import sys
import re
import warnings
import unicodedata
import html
from datetime import datetime

# Core ML imports
import joblib
import pickle
import numpy as np
import pandas as pd

# Defer heavy ML imports to runtime to avoid environment segfaults
tf = None
pad_sequences = None
torch = None
AutoTokenizer = None
AutoModel = None

# Web framework imports
from flask import Flask, request, jsonify
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import json

# Import our custom modules (if available)
try:
    from utils.model_loader import load_models, get_model_info
    from utils.predictor import UnifiedPredictor
    STREAMLIT_UTILS_AVAILABLE = True
except ImportError:
    STREAMLIT_UTILS_AVAILABLE = False
    print("Streamlit utils not available, falling back to Flask-only mode")
# Enhanced Text Preprocessor
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

# Unified Model Predictor Class
class ModelPredictor:
    """Unified predictor for all models"""
    
    def __init__(self):
        self.models = {}
        self.preprocessor = TextPreprocessor()
        self.load_models()
    
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
        
        # Defer loading LSTM/BERT to first use to avoid startup crashes
        print("ℹ️ LSTM and BERT will be loaded lazily on first use")
    
    def wordopt(self, text):
        """Text preprocessing function"""
        text = str(text).lower().strip()
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>+', '', text)
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def predict_svm(self, text):
        """Predict using SVM model"""
        if 'svm' not in self.models:
            return {"error": "SVM model not available"}
        
        cleaned = self.wordopt(text)
        vect = self.models['svm']['vectorizer'].transform([cleaned])
        pred = self.models['svm']['model'].predict(vect)[0]
        conf = self.models['svm']['model'].predict_proba(vect)[0].max()
        label = "Real" if pred == 1 else "Fake"
        return {"label": label, "confidence": round(float(conf), 4)}
    
    def predict_lstm(self, text):
        """Predict using LSTM model"""
        global tf, pad_sequences
        if 'lstm' not in self.models:
            try:
                if tf is None:
                    import tensorflow as tf  # type: ignore
                if pad_sequences is None:
                    from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
                lstm_model = tf.keras.models.load_model('models/lstm_fake_news_model.h5')
                lstm_tokenizer = pickle.load(open('models/lstm_tokenizer.pkl', 'rb'))
                self.models['lstm'] = {
                    'model': lstm_model,
                    'tokenizer': lstm_tokenizer,
                    'accuracy': 0.9890,
                    'type': 'Deep Learning (LSTM)'
                }
                print("✅ LSTM model loaded (lazy)")
            except Exception as e:
                return {"error": f"LSTM model not available: {e}"}
        
        cleaned = self.wordopt(text)
        sequence = self.models['lstm']['tokenizer'].texts_to_sequences([cleaned])
        padded_sequence = pad_sequences(sequence, maxlen=200, padding='post')
        prediction = self.models['lstm']['model'].predict(padded_sequence, verbose=0)[0][0]
        label = "Real" if prediction > 0.5 else "Fake"
        confidence = max(prediction, 1 - prediction)
        return {"label": label, "confidence": round(float(confidence), 4)}
    
    def extract_bert_features(self, texts, batch_size=16):
        """Extract BERT features from texts"""
        if 'bert' not in self.models:
            return None
        
        device = torch.device('cuda' if torch and torch.cuda.is_available() else 'cpu')
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
    
    def predict_bert(self, text):
        """Predict using BERT model"""
        global AutoTokenizer, AutoModel, torch, np
        if 'bert' not in self.models:
            try:
                if AutoTokenizer is None or AutoModel is None:
                    from transformers import AutoTokenizer as _AT, AutoModel as _AM  # type: ignore
                    AutoTokenizer = _AT
                    AutoModel = _AM
                if torch is None:
                    import torch  # type: ignore
                if np is None:
                    import numpy as np  # type: ignore
                bert_classifier = pickle.load(open('models/bert_fake_news_model/classifier.pkl', 'rb'))
                bert_tokenizer = AutoTokenizer.from_pretrained('models/bert_fake_news_model')
                bert_model = AutoModel.from_pretrained('distilbert-base-uncased')
                self.models['bert'] = {
                    'classifier': bert_classifier,
                    'tokenizer': bert_tokenizer,
                    'model': bert_model,
                    'accuracy': 0.9750,
                    'type': 'Transformer (BERT)'
                }
                print("✅ BERT model loaded (lazy)")
            except Exception as e:
                return {"error": f"BERT model not available: {e}"}
        
        cleaned = self.wordopt(text)
        features = self.extract_bert_features([cleaned])
        
        prediction_proba = self.models['bert']['classifier'].predict_proba(features)[0]
        predicted_class = self.models['bert']['classifier'].predict(features)[0]
        confidence = prediction_proba.max()
        
        label = "Real" if predicted_class == 1 else "Fake"
        return {"label": label, "confidence": round(float(confidence), 4)}
    
    def predict_all(self, text):
        """Predict using all available models"""
        results = {}
        
        if 'svm' in self.models:
            results['SVM'] = self.predict_svm(text)
        
        if 'lstm' in self.models:
            results['LSTM'] = self.predict_lstm(text)
        
        if 'bert' in self.models:
            results['BERT'] = self.predict_bert(text)
        
        return results
    
    def get_model_info(self):
        """Get information about all models"""
        info = {
            "available_models": list(self.models.keys()),
            "model_details": {}
        }
        
        descriptions = {
            'svm': 'Linear Support Vector Machine with TF-IDF features',
            'lstm': 'Long Short-Term Memory neural network for sequence modeling',
            'bert': 'Bidirectional Encoder Representations from Transformers'
        }
        
        for name, model_data in self.models.items():
            info["model_details"][name.upper()] = {
                "type": model_data['type'],
                "accuracy": f"{model_data['accuracy']:.2%}",
                "description": descriptions.get(name, 'Unknown model')
            }
        
        return info

# Initialize predictor
predictor = ModelPredictor()

# Flask app initialization
app = Flask(__name__)

# Verdict Agent integration
try:
    from verdict_agent import VerdictAgent, ModelResult
    verdict_agent = VerdictAgent()
    print("✅ Verdict Agent initialized")
except Exception as e:
    verdict_agent = None
    print(f"❌ Verdict Agent unavailable: {e}")

# Flask API Routes
@app.route("/", methods=["GET"])
def home():
    model_info = predictor.get_model_info()
    
    return f"""
    <html>
    <head>
        <title>Advanced Fake News Detection API</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
            .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #333; text-align: center; }}
            .model-card {{ background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; border-left: 4px solid #007bff; }}
            .method {{ color: #28a745; font-weight: bold; }}
            .url {{ color: #007bff; font-family: monospace; }}
            .example {{ background: #e9ecef; padding: 10px; border-radius: 3px; margin: 10px 0; }}
            .status {{ color: #28a745; font-weight: bold; }}
            .accuracy {{ color: #dc3545; font-weight: bold; }}
            .neural {{ color: #6f42c1; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 Advanced Fake News Detection API</h1>
            <p><span class="status">✅ System running with neural network diversity!</span></p>
            
            <h2>Available Models</h2>
            <div class="model-card">
                <h3>🧠 Linear SVM (Traditional ML)</h3>
                <p><span class="accuracy">Accuracy: 99.59%</span></p>
                <p>Fast, lightweight, excellent for real-time predictions</p>
            </div>
            
            <div class="model-card">
                <h3>🔗 LSTM Neural Network</h3>
                <p><span class="accuracy">Accuracy: 98.90%</span></p>
                <p><span class="neural">Deep learning model</span> that captures sequential patterns in text</p>
            </div>
            
            <div class="model-card">
                <h3>🤖 BERT Transformer</h3>
                <p><span class="accuracy">Accuracy: 97.50%</span></p>
                <p><span class="neural">State-of-the-art transformer</span> with contextual understanding</p>
            </div>
            
            <h2>API Endpoints</h2>
            
            <div class="example">
                <h3>Single Model Prediction</h3>
                <span class="method">POST</span> <span class="url">/predict?model=svm</span><br>
                <span class="method">POST</span> <span class="url">/predict?model=lstm</span><br>
                <span class="method">POST</span> <span class="url">/predict?model=bert</span><br>
                Body: <code>{{"text": "Your news article here"}}</code>
            </div>
            
            <div class="example">
                <h3>All Models Comparison</h3>
                <span class="method">POST</span> <span class="url">/predict-all</span><br>
                Body: <code>{{"text": "Your news article here"}}</code>
            </div>
            
            <div class="example">
                <h3>Model Information</h3>
                <span class="method">GET</span> <span class="url">/models</span>
            </div>
            
            <h3>Example Response (All Models)</h3>
            <div class="example">
                <code>
                {{<br>
                &nbsp;&nbsp;"SVM": {{"label": "Real", "confidence": 0.9957}},<br>
                &nbsp;&nbsp;"LSTM": {{"label": "Real", "confidence": 0.9848}},<br>
                &nbsp;&nbsp;"BERT": {{"label": "Real", "confidence": 0.9938}}<br>
                }}
                </code>
            </div>
            
            <h3>Model Performance Comparison</h3>
            <ul>
                <li><strong>SVM:</strong> 99.59% accuracy, fastest inference, traditional ML</li>
                <li><strong>LSTM:</strong> 98.90% accuracy, sequential pattern recognition, neural network</li>
                <li><strong>BERT:</strong> 97.50% accuracy, contextual understanding, transformer</li>
            </ul>
            
            <h3>Neural Network Diversity Achieved! 🎯</h3>
            <p>This system now includes multiple neural network architectures for comprehensive fake news detection.</p>
        </div>
    </body>
    </html>
    """

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")
    model_type = request.args.get("model", "svm").lower()
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    if model_type == "svm":
        result = predictor.predict_svm(text)
    elif model_type == "lstm":
        result = predictor.predict_lstm(text)
    elif model_type == "bert":
        result = predictor.predict_bert(text)
    else:
        return jsonify({"error": f"Unknown model: {model_type}. Available: svm, lstm, bert"}), 400
    
    if "error" in result:
        return jsonify(result), 500
    
    return jsonify(result)

@app.route("/predict-all", methods=["POST"])
def predict_all():
    data = request.json
    text = data.get("text", "")
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    results = predictor.predict_all(text)
    
    if not results:
        return jsonify({"error": "No models available"}), 500
    
    return jsonify(results)

@app.route("/models", methods=["GET"])
def get_models():
    return jsonify(predictor.get_model_info())

@app.route("/health", methods=["GET"])
def health():
    info = predictor.get_model_info()
    return jsonify({
        "status": "ok",
        "available_models": info.get("available_models", [])
    })

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    model_results = predictor.predict_all(text)
    if not model_results:
        return jsonify({"error": "No models available"}), 500
    
    if verdict_agent is None:
        return jsonify({
            "text": text,
            "model_results": model_results,
            "verdict": {"error": "Verdict Agent unavailable"}
        })
    
    # Convert to ModelResult objects
    va_input = {}
    for name, res in model_results.items():
        if "error" not in res:
            key = name.lower()
            model_meta = predictor.models.get(key, {})
            va_input[key] = ModelResult(
                model_name=name,
                label=res.get("label", ""),
                confidence=float(res.get("confidence", 0.0)),
                model_type=model_meta.get("type", "Unknown"),
                accuracy=float(model_meta.get("accuracy", 0.0))
            )
    response = verdict_agent.process_verdict(text, va_input)
    return jsonify({
        "text": text,
        "model_results": model_results,
        "verdict": {
            "verdict": response.verdict.value,
            "confidence": response.confidence,
            "confidence_level": response.confidence_level.value,
            "reasoning": response.reasoning,
            "evidence": [
                {
                    "source": ev.source,
                    "content": ev.content,
                    "relevance_score": ev.relevance_score,
                    "citation": ev.citation
                } for ev in response.evidence
            ],
            "model_agreement": response.model_agreement,
            "audit_id": response.audit_id,
            "timestamp": response.timestamp,
            "processing_time_ms": response.processing_time_ms,
            "explainability": response.explainability
        }
    })

@app.route("/verdict", methods=["POST"])
def verdict_only():
    if verdict_agent is None:
        return jsonify({"error": "Verdict Agent unavailable"}), 500
    data = request.json
    text = data.get("text", "").strip()
    provided = data.get("model_results", {})
    if not text:
        return jsonify({"error": "Text is required"}), 400
    if not provided:
        return jsonify({"error": "Model results are required"}), 400
    
    va_input = {}
    for name, res in provided.items():
        try:
            va_input[name.lower()] = ModelResult(
                model_name=name,
                label=res.get("label", ""),
                confidence=float(res.get("confidence", 0.0)),
                model_type=res.get("model_type", "Unknown"),
                accuracy=float(res.get("accuracy", 0.0))
            )
        except Exception as e:
            return jsonify({"error": f"Invalid model result for {name}: {e}"}), 400
    response = verdict_agent.process_verdict(text, va_input)
    return jsonify({
        "verdict": response.verdict.value,
        "confidence": response.confidence,
        "confidence_level": response.confidence_level.value,
        "reasoning": response.reasoning,
        "evidence": [
            {
                "source": ev.source,
                "content": ev.content,
                "relevance_score": ev.relevance_score,
                "citation": ev.citation
            } for ev in response.evidence
        ],
        "model_agreement": response.model_agreement,
        "audit_id": response.audit_id,
        "timestamp": response.timestamp,
        "processing_time_ms": response.processing_time_ms,
        "explainability": response.explainability
    })

# Streamlit UI Functions (only if utils are available)
def get_model_info():
    """Get model information for Streamlit UI"""
    if not STREAMLIT_UTILS_AVAILABLE:
        # Fallback to basic info
        return {
            'svm': {
                'name': 'SVM',
                'accuracy': 99.5,
                'description': 'Linear Support Vector Machine',
                'strength': 'Fast inference, high accuracy'
            },
            'lstm': {
                'name': 'LSTM',
                'accuracy': 87.0,
                'description': 'Long Short-Term Memory Network',
                'strength': 'Sequential pattern recognition'
            },
            'bert': {
                'name': 'BERT',
                'accuracy': 75.0,
                'description': 'Bidirectional Encoder Representations',
                'strength': 'Contextual understanding'
            }
        }
    return get_model_info()

# Streamlit UI Functions
def streamlit_ui():
    """Streamlit UI interface (only runs if utils are available)"""
    if not STREAMLIT_UTILS_AVAILABLE:
        st.error("Streamlit UI requires utils modules. Please ensure utils/model_loader.py and utils/predictor.py are available.")
        return
    
    # Page configuration
    st.set_page_config(
        page_title="Fake News Detection",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
        .main-header {
            font-size: 3rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .model-card {
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 1rem;
            margin: 1rem 0;
            background-color: #f9f9f9;
        }
        .prediction-result {
            border-radius: 10px;
            padding: 1rem;
            margin: 1rem 0;
        }
        .fake-result {
            background-color: #ffebee;
            border-left: 5px solid #f44336;
        }
        .true-result {
            background-color: #e8f5e8;
            border-left: 5px solid #4caf50;
        }
        .uncertain-result {
            background-color: #fff3e0;
            border-left: 5px solid #ff9800;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)

def initialize_streamlit_session():
    """Initialize Streamlit session state"""
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False

# Load models once (Streamlit only)
@st.cache_resource
def initialize_models():
    """Initialize models and predictor"""
    if not STREAMLIT_UTILS_AVAILABLE:
        return None, None
    model_loader = load_models()
    if model_loader:
        predictor = UnifiedPredictor(model_loader)
        return model_loader, predictor
    return None, None

def main_streamlit():
    """Main Streamlit application"""
    if not STREAMLIT_UTILS_AVAILABLE:
        st.error("Streamlit UI requires utils modules. Please ensure utils/model_loader.py and utils/predictor.py are available.")
        return
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Fake News Detection System</h1>', unsafe_allow_html=True)
    
    # Initialize models
    model_loader, predictor = initialize_models()
    
    if model_loader is None:
        st.error("❌ Failed to load models. Please check that model files are present in the 'models' directory.")
        st.stop()
    
    st.session_state.models_loaded = True
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Main Analysis", "📊 Model Comparison", "📈 Statistics & Insights", "ℹ️ About"]
    )
    
    # Page routing
    if page == "🏠 Main Analysis":
        main_analysis_page(predictor)
    elif page == "📊 Model Comparison":
        model_comparison_page()
    elif page == "📈 Statistics & Insights":
        statistics_page()
    elif page == "ℹ️ About":
        about_page()

def main_analysis_page(predictor):
    """Main prediction interface"""
    st.header("📝 Text Analysis")
    st.markdown("Enter the news text you want to analyze for authenticity.")
    
    # Input form
    with st.form("analysis_form"):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            text = st.text_area(
                "News Text:",
                placeholder="Enter the news text here...",
                height=150,
                help="Minimum 10 characters, maximum 5000 characters"
            )
        
        with col2:
            title = st.text_input("Title (optional):", placeholder="News title")
            source = st.text_input("Source (optional):", placeholder="News source")
        
        analyze_button = st.form_submit_button("🔍 Analyze News", type="primary")
    
    if analyze_button:
        if not text or len(text.strip()) < 10:
            st.error("Please enter at least 10 characters of text.")
        else:
            with st.spinner("Analyzing text..."):
                # Perform analysis using the unified predictor
                results = predictor.predict_all(text)
                
                # Display results
                display_analysis_results(results, text)

def display_analysis_results(results, text):
    """Display comprehensive analysis results"""
    st.header("🎯 Analysis Results")
    
    # Display individual model predictions
    for model_name, result in results.items():
        if "error" not in result:
            with st.expander(f"{model_name} Model Results"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Prediction", result['label'])
                
                with col2:
                    st.metric("Confidence", f"{result['confidence']:.1%}")

def model_comparison_page():
    """Model comparison and performance metrics"""
    st.header("📊 Model Comparison")
    
    model_info = get_model_info()
    
    # Performance metrics table
    st.subheader("Performance Metrics")
    
    metrics_data = []
    for model_key, info in model_info.items():
        metrics_data.append({
            'Model': info['name'],
            'Accuracy (%)': info['accuracy'],
            'Type': info['description'],
            'Strength': info['strength']
        })
    
    df = pd.DataFrame(metrics_data)
    st.dataframe(df, use_container_width=True)

def statistics_page():
    """Statistics and insights from analysis history"""
    st.header("📈 Statistics & Insights")
    
    if not st.session_state.analysis_history:
        st.info("No analysis history available. Perform some analyses to see statistics here.")
        return
    
    st.info("Statistics functionality would be implemented here with the full Streamlit UI.")

def about_page():
    """About page with project information"""
    st.header("ℹ️ About This Project")
    
    st.markdown("""
    ## 🔍 Fake News Detection System
    
    This application uses advanced machine learning models to analyze and detect fake news with high accuracy.
    
    ### 🎯 Features
    
    - **Multi-Model Analysis**: Combines SVM, LSTM, and BERT models for robust detection
    - **Real-time Analysis**: Get instant results with detailed explanations
    - **Credibility Assessment**: Advanced credibility analysis with risk factor identification
    - **Model Comparison**: Compare performance of different models
    - **Statistical Insights**: Track analysis patterns and model performance
    
    ### 🤖 Model Architecture
    
    #### 1. Support Vector Machine (SVM)
    - **Accuracy**: 99.5%
    - **Type**: Traditional machine learning with TF-IDF features
    - **Strength**: Excellent performance on structured text features
    
    #### 2. Long Short-Term Memory (LSTM)
    - **Accuracy**: 87.0%
    - **Type**: Deep learning for sequential data
    - **Strength**: Captures temporal patterns and context flow
    
    #### 3. DistilBERT
    - **Accuracy**: 75.0%
    - **Type**: Transformer model with attention mechanism
    - **Strength**: State-of-the-art text understanding and semantics
    
    ### 🔧 Technical Implementation
    
    - **Frontend**: Streamlit for interactive web interface
    - **Backend**: Python with TensorFlow, PyTorch, and scikit-learn
    - **Deployment**: Hugging Face Spaces for easy access
    - **Memory Optimization**: BERT model uses half-precision for efficiency
    
    ### ⚠️ Important Limitations
    
    - This tool is for educational and research purposes
    - Results should not be the sole basis for important decisions
    - Models may have biases based on training data
    - Always verify information through multiple reliable sources
    
    ### 📚 Responsible Use
    
    - Use as a supplementary tool for fact-checking
    - Combine with human judgment and additional verification
    - Be aware of potential biases in automated systems
    - Respect privacy and ethical considerations
    
    ### 🚀 Deployment
    
    This application is deployed on Hugging Face Spaces, making it accessible to users worldwide.
    
    ### 📞 Support
    
    For questions or issues, please refer to the GitHub repository or Hugging Face Space.
    
    ---
    
    **Built with ❤️ using Streamlit, TensorFlow, PyTorch, and Hugging Face**
    """)

# Main execution logic
def main():
    """Main application entry point"""
    # Check if running as Streamlit app
    try:
        # This will work if running with Streamlit
        if hasattr(st, 'run'):
            initialize_streamlit_session()
            main_streamlit()
            return
    except:
        pass
    
    # Default to Flask API mode
    print("\n🚀 Starting Advanced Fake News Detection API...")
    print("Available endpoints:")
    print("  GET  / - Homepage with documentation")
    print("  GET  /health - Health check")
    print("  POST /predict?model=svm - SVM prediction")
    print("  POST /predict?model=lstm - LSTM prediction")
    print("  POST /predict?model=bert - BERT prediction")
    print("  POST /predict-all - All models prediction")
    print("  GET  /models - Model information")
    print("\nStarting server on http://localhost:5000")
    app.run(debug=False, use_reloader=False, host='127.0.0.1', port=5000)

if __name__ == "__main__":
    main()