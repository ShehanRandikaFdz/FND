"""
Credibility Analyzer for Flask Application
Advanced ensemble methods with confidence calibration
"""

import numpy as np
from typing import Dict, List, Tuple
from .text_preprocessor import TextPreprocessor
from .confidence_calibrator import ConfidenceCalibrator
from .feature_extractor import FeatureExtractor

class CredibilityAnalyzer:
    """Advanced credibility analysis with ensemble methods"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.text_preprocessor = TextPreprocessor()
        self.confidence_calibrator = ConfidenceCalibrator()
        self.feature_extractor = FeatureExtractor()
        
        # Model weights based on performance
        self.model_weights = {
            'svm': 0.4,
            'lstm': 0.3,
            'bert': 0.3
        }
    
    def analyze(self, text: str) -> Dict:
        """Perform comprehensive credibility analysis"""
        try:
            # Preprocess text
            processed_text = self.text_preprocessor.clean_text(text)
            
            # Extract features
            features = self.feature_extractor.extract_features(processed_text)
            
            # Calculate credibility score
            credibility_score = self._calculate_credibility_score(features)
            
            # Estimate uncertainty
            uncertainty = self._estimate_uncertainty(features)
            
            # Identify risk factors
            risk_factors = self._identify_risk_factors(processed_text, features)
            
            return {
                'credibility_score': credibility_score,
                'uncertainty': uncertainty,
                'risk_factors': risk_factors,
                'features': features,
                'processed_text': processed_text
            }
            
        except Exception as e:
            return {
                'credibility_score': 0.5,
                'uncertainty': 1.0,
                'risk_factors': [f'Analysis error: {str(e)}'],
                'features': {},
                'error': str(e)
            }
    
    def _calculate_credibility_score(self, features: Dict) -> float:
        """Calculate overall credibility score from features"""
        # Base score
        base_score = 0.5
        
        # Adjust based on text quality indicators
        if features.get('readability_score', 0) > 0.7:
            base_score += 0.1
        
        if features.get('sentiment_bias', 0) < 0.3:
            base_score += 0.1
        
        if features.get('factual_indicators', 0) > 0.5:
            base_score += 0.2
        
        # Penalize for sensational language
        if features.get('sensational_score', 0) > 0.7:
            base_score -= 0.2
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, base_score))
    
    def _estimate_uncertainty(self, features: Dict) -> float:
        """Estimate uncertainty in the analysis"""
        # Higher uncertainty for shorter texts
        text_length_factor = max(0.1, 1.0 - features.get('normalized_length', 0))
        
        # Higher uncertainty for extreme sentiment
        sentiment_factor = abs(features.get('sentiment_score', 0) - 0.5) * 2
        
        # Combine factors
        uncertainty = (text_length_factor + sentiment_factor) / 2
        return min(1.0, uncertainty)
    
    def _identify_risk_factors(self, text: str, features: Dict) -> List[str]:
        """Identify potential risk factors"""
        risk_factors = []
        
        # Text length
        if features.get('normalized_length', 1) < 0.3:
            risk_factors.append("Very short text")
        
        # Sensational language
        if features.get('sensational_score', 0) > 0.7:
            risk_factors.append("Sensational language detected")
        
        # Sentiment bias
        if features.get('sentiment_bias', 0) > 0.7:
            risk_factors.append("High emotional bias")
        
        # Readability
        if features.get('readability_score', 1) < 0.4:
            risk_factors.append("Poor readability")
        
        return risk_factors
