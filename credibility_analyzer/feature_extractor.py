"""
Feature Extractor for Credibility Analyzer
"""

import re
from typing import Dict, List

class FeatureExtractor:
    """Extract features for credibility analysis"""
    
    def __init__(self):
        self.sensational_words = [
            'shocking', 'breaking', 'exclusive', 'unbelievable', 'amazing', 'incredible',
            'stunning', 'devastating', 'outrageous', 'scandalous', 'explosive', 'bombshell'
        ]
        
        self.factual_indicators = [
            'according to', 'study shows', 'research indicates', 'official', 'government',
            'university', 'scientists', 'peer-reviewed', 'journal', 'published', 'verified',
            'confirmed', 'data shows', 'statistics', 'survey', 'report'
        ]
    
    def extract_features(self, text: str) -> Dict:
        """Extract comprehensive features from text"""
        features = {}
        
        # Basic text features
        features['word_count'] = len(text.split())
        features['char_count'] = len(text)
        features['normalized_length'] = min(1.0, features['word_count'] / 200)
        
        # Sentiment features
        sentiment_score = self._calculate_sentiment(text)
        features['sentiment_score'] = sentiment_score
        features['sentiment_bias'] = abs(sentiment_score - 0.5) * 2
        
        # Readability features
        features['readability_score'] = self._calculate_readability(text)
        
        # Sensational language
        features['sensational_score'] = self._calculate_sensational_score(text)
        
        # Factual indicators
        features['factual_indicators'] = self._count_factual_indicators(text)
        
        # Punctuation analysis
        features['exclamation_ratio'] = text.count('!') / max(1, features['word_count'])
        features['question_ratio'] = text.count('?') / max(1, features['word_count'])
        
        return features
    
    def _calculate_sentiment(self, text: str) -> float:
        """Simple sentiment calculation"""
        positive_words = ['good', 'great', 'excellent', 'positive', 'success', 'win', 'achievement']
        negative_words = ['bad', 'terrible', 'awful', 'negative', 'failure', 'lose', 'problem']
        
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words == 0:
            return 0.5  # Neutral
        
        return positive_count / total_sentiment_words
    
    def _calculate_readability(self, text: str) -> float:
        """Simple readability score"""
        sentences = re.split(r'[.!?]+', text)
        words = text.split()
        
        if not sentences or not words:
            return 0.5
        
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Simple readability (higher is better, normalized to 0-1)
        readability = 1.0 / (1.0 + (avg_sentence_length / 20) + (avg_word_length / 8))
        return min(1.0, readability)
    
    def _calculate_sensational_score(self, text: str) -> float:
        """Calculate sensational language score"""
        text_lower = text.lower()
        sensational_count = sum(1 for word in self.sensational_words if word in text_lower)
        
        # Normalize by text length
        word_count = len(text.split())
        return min(1.0, sensational_count / max(1, word_count / 50))
    
    def _count_factual_indicators(self, text: str) -> float:
        """Count factual indicators in text"""
        text_lower = text.lower()
        factual_count = sum(1 for phrase in self.factual_indicators if phrase in text_lower)
        
        # Normalize by text length
        word_count = len(text.split())
        return min(1.0, factual_count / max(1, word_count / 100))
