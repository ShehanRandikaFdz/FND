"""
Fallback Predictor for when models fail to load
Provides basic text analysis capabilities without ML models
"""

import re
import string
from typing import Dict, List

class FallbackPredictor:
    """Simple rule-based predictor when ML models are unavailable"""
    
    def __init__(self):
        self.fake_indicators = [
            'breaking', 'shocking', 'you won\'t believe', 'doctors hate',
            'click here', 'share this', 'viral', 'amazing discovery',
            'secret', 'hidden truth', 'mainstream media won\'t tell',
            'conspiracy', 'cover-up', 'exposed', 'leaked'
        ]
        
        self.credibility_indicators = [
            'according to', 'study shows', 'research indicates',
            'peer-reviewed', 'published in', 'journal of',
            'university', 'institute', 'department of'
        ]
        
        self.emotional_words = [
            'outrageous', 'incredible', 'unbelievable', 'shocking',
            'terrifying', 'amazing', 'incredible', 'mind-blowing'
        ]
    
    def analyze_text(self, text: str) -> Dict:
        """Analyze text using simple heuristics"""
        text_lower = text.lower()
        
        # Count indicators
        fake_score = 0
        credibility_score = 0
        emotional_score = 0
        
        # Check for fake indicators
        for indicator in self.fake_indicators:
            if indicator in text_lower:
                fake_score += 1
        
        # Check for credibility indicators
        for indicator in self.credibility_indicators:
            if indicator in text_lower:
                credibility_score += 1
        
        # Check for emotional language
        for word in self.emotional_words:
            if word in text_lower:
                emotional_score += 1
        
        # Check for excessive capitalization
        caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        if caps_ratio > 0.3:
            fake_score += 2
        
        # Check for excessive punctuation
        punct_count = sum(1 for c in text if c in '!?')
        if punct_count > 5:
            fake_score += 1
        
        # Check for suspicious URLs or domains
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, text)
        if len(urls) > 2:
            fake_score += 1
        
        # Calculate scores
        total_indicators = len(self.fake_indicators) + len(self.credibility_indicators) + len(self.emotional_words)
        fake_percentage = (fake_score / total_indicators) * 100
        credibility_percentage = (credibility_score / total_indicators) * 100
        
        # Determine prediction
        if fake_score > credibility_score:
            prediction = "FAKE"
            confidence = min(85, fake_percentage + 20)
        elif credibility_score > fake_score:
            prediction = "TRUE"
            confidence = min(85, credibility_percentage + 20)
        else:
            prediction = "TRUE"  # Default to true when uncertain
            confidence = 60
        
        return {
            "final_prediction": prediction,
            "confidence": round(confidence, 2),
            "votes": {"FAKE": 1 if prediction == "FAKE" else 0, "TRUE": 1 if prediction == "TRUE" else 0},
            "individual_results": {
                "fallback": {
                    "prediction": prediction,
                    "confidence": confidence,
                    "model_name": "Fallback Analyzer"
                }
            },
            "total_models": 1,
            "majority_rule": True,
            "analysis_details": {
                "fake_indicators_found": fake_score,
                "credibility_indicators_found": credibility_score,
                "emotional_language_score": emotional_score,
                "caps_ratio": round(caps_ratio, 3),
                "suspicious_urls": len(urls)
            }
        }
