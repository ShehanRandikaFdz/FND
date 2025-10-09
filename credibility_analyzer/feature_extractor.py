"""
Advanced Feature Extraction for Credibility Analysis
Extracts comprehensive features beyond just text content.
"""

import re
import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, List, Tuple, Optional
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import datetime
from textstat import flesch_reading_ease, flesch_kincaid_grade
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

class AdvancedFeatureExtractor:
    """
    Extracts comprehensive features for credibility analysis.
    """
    
    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        
        # Credibility indicators
        self.credible_sources = {
            'reuters', 'ap', 'associated press', 'bbc', 'cnn', 'nbc', 'abc', 'cbs',
            'npr', 'pbs', 'washington post', 'new york times', 'wall street journal',
            'guardian', 'bloomberg', 'financial times', 'usa today'
        }
        
        self.fake_indicators = {
            'breaking', 'urgent', 'shocking', 'unbelievable', 'incredible', 'amazing',
            'you won\'t believe', 'doctors hate', 'one weird trick', 'click here',
            'must see', 'must read', 'what happens next', 'this will shock you'
        }
        
        # Emotional manipulation words
        self.emotional_words = {
            'outrage', 'scandal', 'crisis', 'disaster', 'catastrophe', 'terror',
            'fear', 'panic', 'fury', 'rage', 'explosive', 'bombshell'
        }
        
        # Uncertainty/hedge words
        self.uncertainty_words = {
            'allegedly', 'reportedly', 'supposedly', 'apparently', 'seemingly',
            'possibly', 'probably', 'might', 'could', 'may', 'perhaps'
        }
        
        # Authority/expertise indicators
        self.authority_words = {
            'expert', 'professor', 'doctor', 'researcher', 'scientist', 'analyst',
            'official', 'spokesperson', 'according to', 'study shows', 'research indicates'
        }
    
    def extract_basic_features(self, text: str) -> Dict[str, float]:
        """Extract basic text statistics."""
        words = word_tokenize(text.lower())
        sentences = sent_tokenize(text)
        
        features = {
            'char_count': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
            'unique_word_ratio': len(set(words)) / len(words) if words else 0,
        }
        
        return features
    
    def extract_punctuation_features(self, text: str) -> Dict[str, float]:
        """Extract punctuation-based features."""
        features = {
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'period_count': text.count('.'),
            'comma_count': text.count(','),
            'semicolon_count': text.count(';'),
            'colon_count': text.count(':'),
            'quotation_count': text.count('"') + text.count("'"),
            'ellipsis_count': text.count('...'),
        }
        
        # Normalize by text length
        text_length = max(len(text), 1)
        normalized_features = {}
        for key, value in features.items():
            normalized_features[f'{key}_normalized'] = value / text_length
        features.update(normalized_features)
        
        return features
    
    def extract_capitalization_features(self, text: str) -> Dict[str, float]:
        """Extract capitalization patterns."""
        words = text.split()
        
        features = {
            'caps_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'all_caps_words': sum(1 for word in words if word.isupper() and len(word) > 1),
            'title_case_words': sum(1 for word in words if word.istitle()),
            'mixed_case_words': sum(1 for word in words if any(c.isupper() for c in word) and any(c.islower() for c in word)),
        }
        
        # Normalize by word count
        word_count = max(len(words), 1)
        features['all_caps_ratio'] = features['all_caps_words'] / word_count
        features['title_case_ratio'] = features['title_case_words'] / word_count
        features['mixed_case_ratio'] = features['mixed_case_words'] / word_count
        
        return features
    
    def extract_url_and_link_features(self, text: str) -> Dict[str, float]:
        """Extract URL and link-related features."""
        url_patterns = [
            r'https?://[^\s]+',
            r'www\.[^\s]+',
            r'bit\.ly/[^\s]+',
            r'tinyurl\.com/[^\s]+',
            r'[^\s]+\.com[^\s]*',
            r'[^\s]+\.org[^\s]*',
            r'[^\s]+\.net[^\s]*'
        ]
        
        features = {
            'url_count': 0,
            'short_url_count': 0,
            'suspicious_domain_count': 0,
        }
        
        for pattern in url_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            features['url_count'] += len(matches)
            
            # Check for URL shorteners
            short_url_patterns = ['bit.ly', 'tinyurl', 't.co', 'goo.gl']
            for match in matches:
                if any(short in match.lower() for short in short_url_patterns):
                    features['short_url_count'] += 1
        
        # Suspicious domains (common in fake news)
        suspicious_domains = ['.tk', '.ml', '.ga', '.cf', 'blogspot', 'wordpress']
        for domain in suspicious_domains:
            features['suspicious_domain_count'] += len(re.findall(domain, text, re.IGNORECASE))
        
        return features
    
    def extract_sentiment_features(self, text: str) -> Dict[str, float]:
        """Extract sentiment and emotional features."""
        sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
        
        features = {
            'sentiment_positive': sentiment_scores['pos'],
            'sentiment_negative': sentiment_scores['neg'],
            'sentiment_neutral': sentiment_scores['neu'],
            'sentiment_compound': sentiment_scores['compound'],
        }
        
        # Emotional intensity
        text_lower = text.lower()
        features['emotional_word_count'] = sum(
            text_lower.count(word) for word in self.emotional_words
        )
        
        return features
    
    def extract_credibility_indicators(self, text: str) -> Dict[str, float]:
        """Extract features indicating credibility or lack thereof."""
        text_lower = text.lower()
        
        features = {
            'credible_source_mentions': 0,
            'fake_indicator_count': 0,
            'uncertainty_word_count': 0,
            'authority_word_count': 0,
        }
        
        # Count credible source mentions
        for source in self.credible_sources:
            if source in text_lower:
                features['credible_source_mentions'] += 1
        
        # Count fake news indicators
        for indicator in self.fake_indicators:
            features['fake_indicator_count'] += text_lower.count(indicator)
        
        # Count uncertainty words
        for word in self.uncertainty_words:
            features['uncertainty_word_count'] += text_lower.count(word)
        
        # Count authority indicators
        for word in self.authority_words:
            features['authority_word_count'] += text_lower.count(word)
        
        return features
    
    def extract_readability_features(self, text: str) -> Dict[str, float]:
        """Extract readability and complexity features."""
        try:
            features = {
                'flesch_reading_ease': flesch_reading_ease(text),
                'flesch_kincaid_grade': flesch_kincaid_grade(text),
            }
        except:
            features = {
                'flesch_reading_ease': 0.0,
                'flesch_kincaid_grade': 0.0,
            }
        
        # Additional complexity measures
        words = word_tokenize(text.lower())
        sentences = sent_tokenize(text)
        
        if words and sentences:
            # Lexical diversity
            features['lexical_diversity'] = len(set(words)) / len(words)
            
            # Average syllables per word (approximation)
            syllable_count = sum(max(1, len(re.findall(r'[aeiouAEIOU]', word))) for word in words)
            features['avg_syllables_per_word'] = syllable_count / len(words)
            
            # Complex word ratio (words with 3+ syllables)
            complex_words = sum(1 for word in words if len(re.findall(r'[aeiouAEIOU]', word)) >= 3)
            features['complex_word_ratio'] = complex_words / len(words)
        else:
            features.update({
                'lexical_diversity': 0.0,
                'avg_syllables_per_word': 0.0,
                'complex_word_ratio': 0.0
            })
        
        return features
    
    def extract_temporal_features(self, text: str) -> Dict[str, float]:
        """Extract temporal and urgency features."""
        text_lower = text.lower()
        
        # Time-related words
        time_words = ['today', 'yesterday', 'tomorrow', 'now', 'immediately', 'urgent', 'breaking']
        temporal_count = sum(text_lower.count(word) for word in time_words)
        
        # Date patterns
        date_patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # MM/DD/YYYY
            r'\b\d{1,2}-\d{1,2}-\d{2,4}\b',  # MM-DD-YYYY
            r'\b\w+ \d{1,2}, \d{4}\b',       # Month DD, YYYY
        ]
        
        date_count = sum(len(re.findall(pattern, text)) for pattern in date_patterns)
        
        features = {
            'temporal_word_count': temporal_count,
            'date_mention_count': date_count,
            'urgency_score': text_lower.count('urgent') + text_lower.count('breaking') + text_lower.count('now'),
        }
        
        return features
    
    def extract_linguistic_features(self, text: str) -> Dict[str, float]:
        """Extract linguistic and stylistic features."""
        words = word_tokenize(text.lower())
        
        # Part-of-speech distribution (simplified)
        features = {
            'pronoun_count': 0,
            'adjective_count': 0,
            'adverb_count': 0,
            'modal_verb_count': 0,
        }
        
        # Simple heuristics for POS (not perfect but fast)
        pronouns = {'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        modal_verbs = {'can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would'}
        
        for word in words:
            if word in pronouns:
                features['pronoun_count'] += 1
            elif word in modal_verbs:
                features['modal_verb_count'] += 1
            elif word.endswith('ly'):
                features['adverb_count'] += 1
            elif word.endswith(('ive', 'ous', 'ful', 'less', 'able')):
                features['adjective_count'] += 1
        
        # Normalize by word count
        word_count = max(len(words), 1)
        for key in ['pronoun_count', 'adjective_count', 'adverb_count', 'modal_verb_count']:
            features[f'{key}_ratio'] = features[key] / word_count
        
        return features
    
    def extract_all_features(self, text: str) -> Dict[str, float]:
        """Extract all features and combine them."""
        all_features = {}
        
        # Extract different feature groups
        feature_extractors = [
            self.extract_basic_features,
            self.extract_punctuation_features,
            self.extract_capitalization_features,
            self.extract_url_and_link_features,
            self.extract_sentiment_features,
            self.extract_credibility_indicators,
            self.extract_readability_features,
            self.extract_temporal_features,
            self.extract_linguistic_features,
        ]
        
        for extractor in feature_extractors:
            try:
                features = extractor(text)
                all_features.update(features)
            except Exception as e:
                print(f"Warning: Feature extraction failed for {extractor.__name__}: {e}")
        
        return all_features
    
    def get_feature_importance_score(self, features: Dict[str, float]) -> float:
        """Calculate a composite credibility score based on features."""
        # Weights based on empirical importance for fake news detection
        weights = {
            'credible_source_mentions': 0.3,
            'fake_indicator_count': -0.4,
            'authority_word_count': 0.2,
            'uncertainty_word_count': -0.1,
            'emotional_word_count': -0.2,
            'exclamation_count_normalized': -0.15,
            'all_caps_ratio': -0.1,
            'flesch_reading_ease': 0.1,
            'lexical_diversity': 0.1,
            'urgency_score': -0.2,
        }
        
        score = 0.5  # Base score (neutral)
        
        for feature, weight in weights.items():
            if feature in features:
                score += weight * features[feature]
        
        # Normalize to [0, 1]
        return max(0.0, min(1.0, score))

def main():
    """Test the advanced feature extractor."""
    print("🔍 Testing Advanced Feature Extractor")
    print("=" * 50)
    
    extractor = AdvancedFeatureExtractor()
    
    test_texts = [
        "Reuters reports that the Federal Reserve announced new interest rate policies today after careful analysis.",
        "BREAKING!!! You WON'T believe what scientists discovered! Doctors HATE this one weird trick! Click here NOW!",
        "According to a study published in Nature, researchers have found evidence of water on Mars.",
        "URGENT: This shocking discovery will change everything! Government officials don't want you to know!",
        "The President signed the new healthcare bill today, according to official White House sources."
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n📰 Test {i}: {text[:60]}...")
        print("-" * 70)
        
        features = extractor.extract_all_features(text)
        credibility_score = extractor.get_feature_importance_score(features)
        
        print(f"🎯 Credibility Score: {credibility_score:.4f}")
        
        # Show key features
        key_features = [
            'credible_source_mentions', 'fake_indicator_count', 'authority_word_count',
            'emotional_word_count', 'exclamation_count', 'all_caps_ratio',
            'sentiment_compound', 'flesch_reading_ease', 'urgency_score'
        ]
        
        print("📊 Key Features:")
        for feature in key_features:
            if feature in features:
                print(f"   {feature}: {features[feature]:.3f}")
        
        print(f"📈 Total features extracted: {len(features)}")

if __name__ == "__main__":
    main()
