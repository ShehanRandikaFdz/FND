"""
Text Preprocessor for Credibility Analyzer
"""

import re
import string
from typing import Dict

class TextPreprocessor:
    """Text preprocessing for credibility analysis"""
    
    def __init__(self):
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these',
            'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove excessive punctuation
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        # Remove punctuation and split
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = text.split()
        
        # Remove stop words
        tokens = [token for token in tokens if token not in self.stop_words]
        
        return tokens
    
    def get_word_count(self, text: str) -> int:
        """Get word count"""
        return len(self.tokenize(text))
    
    def get_sentence_count(self, text: str) -> int:
        """Get sentence count"""
        sentences = re.split(r'[.!?]+', text)
        return len([s for s in sentences if s.strip()])
    
    def get_avg_word_length(self, text: str) -> float:
        """Get average word length"""
        tokens = self.tokenize(text)
        if not tokens:
            return 0.0
        return sum(len(token) for token in tokens) / len(tokens)
