"""
Final Enhanced Fake News Detection API with SVM, LSTM, and BERT Models
Complete implementation with all neural network diversity.
"""

from flask import Flask, request, jsonify
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
warnings.filterwarnings('ignore')

# Load models
print("Loading all models...")

class ModelPredictor:
    """Unified predictor for all models"""
    
    def __init__(self):
        self.models = {}
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
        
        # Load LSTM
        try:
            self.models['lstm'] = {
                'model': tf.keras.models.load_model('models/lstm_fake_news_model.h5'),
                'tokenizer': pickle.load(open('models/lstm_tokenizer.pkl', 'rb')),
                'accuracy': 0.9890,
                'type': 'Deep Learning (LSTM)'
            }
            print("✅ LSTM model loaded")
        except Exception as e:
            print(f"❌ Error loading LSTM model: {e}")
        
        # Load BERT
        try:
            self.models['bert'] = {
                'classifier': pickle.load(open('models/bert_fake_news_model/classifier.pkl', 'rb')),
                'tokenizer': AutoTokenizer.from_pretrained('models/bert_fake_news_model'),
                'model': AutoModel.from_pretrained('distilbert-base-uncased'),
                'accuracy': 0.9750,
                'type': 'Transformer (BERT)'
            }
            print("✅ BERT model loaded")
        except Exception as e:
            print(f"❌ Error loading BERT model: {e}")
    
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
        if 'lstm' not in self.models:
            return {"error": "LSTM model not available"}
        
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
    
    def predict_bert(self, text):
        """Predict using BERT model"""
        if 'bert' not in self.models:
            return {"error": "BERT model not available"}
        
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

# Flask app
app = Flask(__name__)

# Homepage route
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

# Single model prediction endpoint
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

# All models prediction endpoint
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

# Model information endpoint
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

if __name__ == "__main__":
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
    app.run(debug=False, use_reloader=False)
