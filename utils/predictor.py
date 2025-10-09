"""
Unified Prediction Interface
Integrates all three models with credibility_analyzer and verdict_agent
"""

import re
import numpy as np
import torch
from typing import Dict, List, Any, Optional
import streamlit as st

# Import prediction functions from main_three_models.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from main_three_models import preprocess_text, predict_svm, predict_lstm, predict_bert
    from credibility_analyzer.credibility_analyzer import CredibilityAnalyzer
    from verdict_agent.verdict_agent import VerdictAgent, ModelResult, VerdictType
except ImportError as e:
    st.warning(f"Some components not available: {e}")

class UnifiedPredictor:
    """Unified prediction interface for all models and analysis systems"""
    
    def __init__(self, model_loader):
        self.model_loader = model_loader
        self.credibility_analyzer = None
        self.verdict_agent = None
        self._initialize_analysis_systems()
    
    def _initialize_analysis_systems(self):
        """Initialize credibility analyzer and verdict agent"""
        try:
            # Initialize credibility analyzer
            self.credibility_analyzer = CredibilityAnalyzer(models_dir="models")
            st.success("✅ Credibility Analyzer initialized")
        except Exception as e:
            st.warning(f"⚠️ Credibility Analyzer not available: {e}")
        
        try:
            # Initialize verdict agent
            self.verdict_agent = VerdictAgent()
            st.success("✅ Verdict Agent initialized")
        except Exception as e:
            st.warning(f"⚠️ Verdict Agent not available: {e}")
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if not text:
            return ""
        
        # Basic text cleaning
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def predict_svm(self, text: str) -> Dict:
        """Get SVM prediction"""
        try:
            if self.model_loader.models.get('svm') is None:
                return {"prediction": "ERROR", "confidence": 0.0, "error": "SVM model not loaded"}
            
            clean_text = self.preprocess_text(text)
            vectorized = self.model_loader.vectorizers['svm'].transform([clean_text])
            prediction = self.model_loader.models['svm'].predict(vectorized)[0]
            probabilities = self.model_loader.models['svm'].predict_proba(vectorized)[0]
            
            pred_label = "FAKE" if prediction == 1 else "TRUE"
            confidence = max(probabilities) * 100
            fake_prob = probabilities[1] * 100
            true_prob = probabilities[0] * 100
            
            return {
                "model_name": "SVM",
                "prediction": pred_label,
                "confidence": round(confidence, 1),
                "probability_fake": round(fake_prob, 1),
                "probability_true": round(true_prob, 1)
            }
        except Exception as e:
            return {"prediction": "ERROR", "confidence": 0.0, "error": str(e)}
    
    def predict_lstm(self, text: str) -> Dict:
        """Get LSTM prediction"""
        try:
            if self.model_loader.models.get('lstm') is None:
                return {"prediction": "ERROR", "confidence": 0.0, "error": "LSTM model not loaded"}
            
            clean_text = self.preprocess_text(text)
            tokenizer = self.model_loader.tokenizers['lstm']
            
            # Tokenize and pad
            sequence = tokenizer.texts_to_sequences([clean_text])
            padded = np.array(sequence)
            
            # Get prediction
            prediction = self.model_loader.models['lstm'].predict(padded, verbose=0)[0]
            fake_prob = prediction[0] * 100
            true_prob = (1 - prediction[0]) * 100
            
            pred_label = "FAKE" if prediction[0] > 0.5 else "TRUE"
            confidence = max(fake_prob, true_prob)
            
            return {
                "model_name": "LSTM",
                "prediction": pred_label,
                "confidence": round(confidence, 1),
                "probability_fake": round(fake_prob, 1),
                "probability_true": round(true_prob, 1)
            }
        except Exception as e:
            return {"prediction": "ERROR", "confidence": 0.0, "error": str(e)}
    
    def predict_bert(self, text: str) -> Dict:
        """Get BERT prediction with memory optimization"""
        try:
            if self.model_loader.models.get('bert') is None:
                return {"prediction": "ERROR", "confidence": 0.0, "error": "BERT model not loaded"}
            
            clean_text = self.preprocess_text(text)
            tokenizer = self.model_loader.tokenizers['bert']
            
            # Tokenize with optimized settings
            inputs = tokenizer(
                clean_text,
                return_tensors="pt",
                max_length=128,  # Reduced for memory efficiency
                truncation=True,
                padding='max_length'
            )
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model_loader.models['bert'](**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)
            
            # Interpret results (0=TRUE, 1=FAKE)
            true_prob = probabilities[0][0].item() * 100
            fake_prob = probabilities[0][1].item() * 100
            
            pred_label = "FAKE" if fake_prob > 50 else "TRUE"
            confidence = max(fake_prob, true_prob)
            
            return {
                "model_name": "BERT",
                "prediction": pred_label,
                "confidence": round(confidence, 1),
                "probability_fake": round(fake_prob, 1),
                "probability_true": round(true_prob, 1)
            }
        except Exception as e:
            return {"prediction": "ERROR", "confidence": 0.0, "error": str(e)}
    
    def get_ensemble_prediction(self, predictions: List[Dict]) -> Dict:
        """Combine predictions from multiple models"""
        try:
            # Filter out error predictions
            valid_predictions = [p for p in predictions if p.get('prediction') not in ['ERROR', 'UNKNOWN']]
            
            if not valid_predictions:
                return {
                    "overall_prediction": "ERROR",
                    "overall_confidence": 0.0,
                    "ensemble_confidence": 0.0
                }
            
            # Model weights based on performance
            weights = {
                "SVM": 0.4,
                "LSTM": 0.3,
                "BERT": 0.3
            }
            
            # Calculate weighted scores
            fake_score = 0
            true_score = 0
            total_weight = 0
            
            for pred in valid_predictions:
                model_name = pred['model_name']
                weight = weights.get(model_name, 0.33)
                
                if pred['prediction'] == 'FAKE':
                    fake_score += weight * (pred['probability_fake'] / 100)
                else:
                    true_score += weight * (pred['probability_true'] / 100)
                
                total_weight += weight
            
            if total_weight > 0:
                fake_score /= total_weight
                true_score /= total_weight
            
            # Determine overall prediction
            if fake_score > true_score:
                overall_prediction = "FAKE"
                overall_confidence = fake_score * 100
            else:
                overall_prediction = "TRUE"
                overall_confidence = true_score * 100
            
            return {
                "overall_prediction": overall_prediction,
                "overall_confidence": round(overall_confidence, 1),
                "ensemble_confidence": round(max(fake_score, true_score) * 100, 1),
                "fake_score": round(fake_score * 100, 1),
                "true_score": round(true_score * 100, 1)
            }
        except Exception as e:
            return {
                "overall_prediction": "ERROR",
                "overall_confidence": 0.0,
                "ensemble_confidence": 0.0,
                "error": str(e)
            }
    
    def get_credibility_analysis(self, text: str) -> Dict:
        """Get credibility analysis if available"""
        try:
            if self.credibility_analyzer is None:
                return {"available": False, "error": "Credibility analyzer not initialized"}
            
            result = self.credibility_analyzer.analyze(text)
            return {
                "available": True,
                "credibility_score": result.get('credibility_score', 0.0),
                "uncertainty": result.get('uncertainty', 0.0),
                "risk_factors": result.get('risk_factors', [])
            }
        except Exception as e:
            return {"available": False, "error": str(e)}
    
    def get_verdict(self, text: str, predictions: List[Dict]) -> Dict:
        """Get verdict from verdict agent if available"""
        try:
            if self.verdict_agent is None:
                return {"available": False, "error": "Verdict agent not initialized"}
            
            # Convert predictions to ModelResult format
            model_results = []
            for pred in predictions:
                if pred.get('prediction') != 'ERROR':
                    model_results.append(ModelResult(
                        model_name=pred['model_name'],
                        label=pred['prediction'].lower(),
                        confidence=pred['confidence'] / 100,
                        model_type="classification",
                        accuracy=0.95  # Default accuracy
                    ))
            
            if model_results:
                verdict_result = self.verdict_agent.generate_verdict(text, model_results)
                return {
                    "available": True,
                    "verdict": verdict_result.get('verdict', 'unknown'),
                    "confidence": verdict_result.get('confidence', 0.0),
                    "explanation": verdict_result.get('explanation', 'No explanation available')
                }
            else:
                return {"available": False, "error": "No valid model results for verdict"}
        except Exception as e:
            return {"available": False, "error": str(e)}
    
    def analyze_text(self, text: str, title: str = "", source: str = "") -> Dict:
        """Complete analysis pipeline"""
        if not text or len(text.strip()) < 10:
            return {
                "error": "Text must be at least 10 characters long",
                "individual_predictions": [],
                "ensemble_result": {},
                "credibility_analysis": {},
                "verdict": {}
            }
        
        # Get individual predictions
        predictions = []
        
        # SVM prediction
        svm_pred = self.predict_svm(text)
        predictions.append(svm_pred)
        
        # LSTM prediction
        lstm_pred = self.predict_lstm(text)
        predictions.append(lstm_pred)
        
        # BERT prediction
        bert_pred = self.predict_bert(text)
        predictions.append(bert_pred)
        
        # Get ensemble result
        ensemble_result = self.get_ensemble_prediction(predictions)
        
        # Get credibility analysis
        credibility_analysis = self.get_credibility_analysis(text)
        
        # Get verdict
        verdict = self.get_verdict(text, predictions)
        
        # Identify risk factors
        risk_factors = self.identify_risk_factors(text)
        
        return {
            "text": text,
            "title": title,
            "source": source,
            "individual_predictions": predictions,
            "ensemble_result": ensemble_result,
            "credibility_analysis": credibility_analysis,
            "verdict": verdict,
            "risk_factors": risk_factors,
            "analysis_summary": self.generate_analysis_summary(ensemble_result, risk_factors)
        }
    
    def identify_risk_factors(self, text: str) -> List[str]:
        """Identify potential risk factors in the text"""
        risk_factors = []
        
        # Check text length
        if len(text) < 50:
            risk_factors.append("Very short text (less than 50 words)")
        
        # Check for excessive capitalization
        if sum(1 for c in text if c.isupper()) > len(text) * 0.3:
            risk_factors.append("Excessive use of capital letters")
        
        # Check for excessive punctuation
        if text.count('!') > 3 or text.count('?') > 5:
            risk_factors.append("Excessive use of punctuation")
        
        # Check for sensational words
        sensational_words = ['BREAKING', 'SHOCKING', 'UNBELIEVABLE', 'EXPOSED', 'REVEALED']
        if any(word in text.upper() for word in sensational_words):
            risk_factors.append("Use of sensational language")
        
        # Check for emotional language
        emotional_words = ['outraged', 'devastated', 'terrified', 'amazing', 'incredible']
        if any(word in text.lower() for word in emotional_words):
            risk_factors.append("Highly emotional language")
        
        return risk_factors
    
    def generate_analysis_summary(self, ensemble_result: Dict, risk_factors: List[str]) -> str:
        """Generate a summary of the analysis"""
        prediction = ensemble_result.get('overall_prediction', 'UNKNOWN')
        confidence = ensemble_result.get('overall_confidence', 0)
        
        if prediction == 'FAKE':
            if confidence > 80:
                summary = f"This text appears to be FAKE with high confidence ({confidence:.1f}%)."
            else:
                summary = f"This text appears to be FAKE with moderate confidence ({confidence:.1f}%)."
        elif prediction == 'TRUE':
            if confidence > 80:
                summary = f"This text appears to be TRUE with high confidence ({confidence:.1f}%)."
            else:
                summary = f"This text appears to be TRUE with moderate confidence ({confidence:.1f}%)."
        else:
            summary = "Unable to determine the authenticity of this text."
        
        if risk_factors:
            summary += f" Identified {len(risk_factors)} potential risk factors."
        
        return summary
