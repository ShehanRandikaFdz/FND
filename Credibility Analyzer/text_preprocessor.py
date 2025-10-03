"""
Enhanced Text Preprocessing Pipeline
Advanced text cleaning, normalization, and validation for credibility analysis.
"""

import re
import unicodedata
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
import langdetect
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class TextPreprocessor:
    """
    Advanced text preprocessing for credibility analysis.
    """
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # Patterns for cleaning
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+|bit\.ly/\S+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'(\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}')
        self.html_pattern = re.compile(r'<[^>]+>')
        self.bracket_pattern = re.compile(r'\[.*?\]|\(.*?\)')
        self.multiple_spaces = re.compile(r'\s+')
        self.multiple_punctuation = re.compile(r'[!]{2,}|[?]{2,}|[.]{3,}')
        
        # Sensational/clickbait patterns
        self.sensational_patterns = [
            r'\b(breaking|urgent|alert|shocking|unbelievable|incredible|amazing)\b',
            r'\byou (won\'t|wont) believe\b',
            r'\bclick here\b',
            r'\bmust (see|read|watch)\b',
            r'\b(doctors|experts) hate (this|him|her)\b',
            r'\bone (weird|simple|strange) trick\b',
            r'\bwhat happens next\b'
        ]
        
        # Validation thresholds
        self.min_length = 10
        self.max_length = 10000
        self.min_words = 3
        self.max_words = 2000
        
    def detect_language(self, text: str) -> Tuple[str, float]:
        """Detect text language and confidence."""
        try:
            lang = langdetect.detect(text)
            # Get confidence (langdetect doesn't provide confidence directly)
            # Use a simple heuristic based on text length and character distribution
            confidence = min(1.0, len(text) / 100.0)  # Simple confidence estimate
            return lang, confidence
        except:
            return 'unknown', 0.0
    
    def normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters."""
        # Normalize to NFC form
        text = unicodedata.normalize('NFC', text)
        
        # Replace common Unicode variations
        replacements = {
            '"': '"', '"': '"',  # Smart quotes
            ''': "'", ''': "'",  # Smart apostrophes
            '–': '-', '—': '-',  # Em/en dashes
            '…': '...',          # Ellipsis
            '\u00a0': ' ',       # Non-breaking space
            '\u2009': ' ',       # Thin space
            '\u200b': '',        # Zero-width space
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def remove_pii(self, text: str) -> str:
        """Remove personally identifiable information."""
        # Remove emails
        text = self.email_pattern.sub('[EMAIL]', text)
        
        # Remove phone numbers
        text = self.phone_pattern.sub('[PHONE]', text)
        
        # Remove potential SSNs (XXX-XX-XXXX pattern)
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
        
        # Remove credit card patterns
        text = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[CARD]', text)
        
        return text
    
    def clean_html_and_markup(self, text: str) -> str:
        """Remove HTML tags and markup."""
        # Remove HTML tags
        text = self.html_pattern.sub(' ', text)
        
        # Remove common markup patterns
        text = re.sub(r'&[a-zA-Z]+;', ' ', text)  # HTML entities
        text = re.sub(r'&#\d+;', ' ', text)       # Numeric entities
        
        # Remove markdown-style formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
        text = re.sub(r'__(.*?)__', r'\1', text)      # Underline
        
        return text
    
    def handle_contractions(self, text: str) -> str:
        """Expand common contractions."""
        contractions = {
            "won't": "will not", "can't": "cannot", "n't": " not",
            "'re": " are", "'ve": " have", "'ll": " will",
            "'d": " would", "'m": " am", "it's": "it is",
            "that's": "that is", "there's": "there is",
            "here's": "here is", "what's": "what is",
            "where's": "where is", "who's": "who is",
            "how's": "how is", "let's": "let us"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
            text = text.replace(contraction.capitalize(), expansion.capitalize())
        
        return text
    
    def normalize_punctuation(self, text: str) -> str:
        """Normalize excessive punctuation."""
        # Replace multiple exclamation marks
        text = re.sub(r'!{2,}', '!', text)
        
        # Replace multiple question marks
        text = re.sub(r'\?{2,}', '?', text)
        
        # Replace multiple periods (but preserve ellipsis)
        text = re.sub(r'\.{4,}', '...', text)
        
        # Normalize spacing around punctuation
        text = re.sub(r'\s*([,.!?;:])\s*', r'\1 ', text)
        
        return text
    
    def extract_sensational_features(self, text: str) -> Dict[str, int]:
        """Extract features indicating sensational/clickbait content."""
        features = {}
        text_lower = text.lower()
        
        # Count sensational patterns
        sensational_count = 0
        for pattern in self.sensational_patterns:
            matches = len(re.findall(pattern, text_lower))
            sensational_count += matches
        
        features['sensational_terms'] = sensational_count
        features['all_caps_words'] = len(re.findall(r'\b[A-Z]{2,}\b', text))
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['caps_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        
        return features
    
    def validate_text(self, text: str) -> Dict[str, any]:
        """Validate text input and return validation results."""
        validation = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'stats': {}
        }
        
        # Basic stats
        char_count = len(text)
        word_count = len(text.split())
        sent_count = len(sent_tokenize(text))
        
        validation['stats'] = {
            'char_count': char_count,
            'word_count': word_count,
            'sentence_count': sent_count,
            'avg_word_length': sum(len(word) for word in text.split()) / max(word_count, 1),
            'avg_sentence_length': word_count / max(sent_count, 1)
        }
        
        # Length validation
        if char_count < self.min_length:
            validation['is_valid'] = False
            validation['issues'].append(f'Text too short ({char_count} < {self.min_length} chars)')
        
        if char_count > self.max_length:
            validation['is_valid'] = False
            validation['issues'].append(f'Text too long ({char_count} > {self.max_length} chars)')
        
        if word_count < self.min_words:
            validation['is_valid'] = False
            validation['issues'].append(f'Too few words ({word_count} < {self.min_words})')
        
        if word_count > self.max_words:
            validation['is_valid'] = False
            validation['issues'].append(f'Too many words ({word_count} > {self.max_words})')
        
        # Language detection
        lang, lang_conf = self.detect_language(text)
        validation['stats']['language'] = lang
        validation['stats']['language_confidence'] = lang_conf
        
        if lang != 'en':
            validation['warnings'].append(f'Non-English text detected: {lang}')
        
        if lang_conf < 0.5:
            validation['warnings'].append(f'Low language detection confidence: {lang_conf:.2f}')
        
        # Content quality checks
        if validation['stats']['avg_word_length'] < 2:
            validation['warnings'].append('Very short average word length')
        
        if validation['stats']['avg_sentence_length'] > 50:
            validation['warnings'].append('Very long average sentence length')
        
        # Check for potential spam/gibberish
        unique_chars = len(set(text.lower()))
        if unique_chars < 10 and char_count > 50:
            validation['warnings'].append('Low character diversity (possible spam)')
        
        return validation
    
    def preprocess_text(self, text: str, 
                       remove_stopwords: bool = False,
                       apply_stemming: bool = False,
                       preserve_case: bool = False) -> Dict[str, any]:
        """
        Main preprocessing function with comprehensive text cleaning.
        """
        if not isinstance(text, str):
            text = str(text)
        
        original_text = text
        
        # Step 1: Validate input
        validation = self.validate_text(text)
        
        if not validation['is_valid']:
            return {
                'processed_text': text,
                'original_text': original_text,
                'validation': validation,
                'preprocessing_applied': [],
                'error': 'Text validation failed'
            }
        
        preprocessing_steps = []
        
        # Step 2: Unicode normalization
        text = self.normalize_unicode(text)
        preprocessing_steps.append('unicode_normalization')
        
        # Step 3: Remove PII
        text = self.remove_pii(text)
        preprocessing_steps.append('pii_removal')
        
        # Step 4: Clean HTML and markup
        text = self.clean_html_and_markup(text)
        preprocessing_steps.append('html_cleanup')
        
        # Step 5: Remove URLs
        text = self.url_pattern.sub('[URL]', text)
        preprocessing_steps.append('url_replacement')
        
        # Step 6: Handle contractions
        text = self.handle_contractions(text)
        preprocessing_steps.append('contraction_expansion')
        
        # Step 7: Normalize punctuation
        text = self.normalize_punctuation(text)
        preprocessing_steps.append('punctuation_normalization')
        
        # Step 8: Remove brackets and parentheses content
        text = self.bracket_pattern.sub(' ', text)
        preprocessing_steps.append('bracket_removal')
        
        # Step 9: Case normalization (unless preserving case)
        if not preserve_case:
            text = text.lower()
            preprocessing_steps.append('case_normalization')
        
        # Step 10: Remove extra whitespace
        text = self.multiple_spaces.sub(' ', text).strip()
        preprocessing_steps.append('whitespace_normalization')
        
        # Step 11: Tokenization and optional processing
        tokens = word_tokenize(text)
        
        if remove_stopwords:
            tokens = [token for token in tokens if token.lower() not in self.stop_words]
            preprocessing_steps.append('stopword_removal')
        
        if apply_stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]
            preprocessing_steps.append('stemming')
        
        # Reconstruct text
        processed_text = ' '.join(tokens)
        
        # Extract additional features
        sensational_features = self.extract_sensational_features(original_text)
        
        return {
            'processed_text': processed_text,
            'original_text': original_text,
            'tokens': tokens,
            'validation': validation,
            'sensational_features': sensational_features,
            'preprocessing_applied': preprocessing_steps,
            'processing_stats': {
                'original_length': len(original_text),
                'processed_length': len(processed_text),
                'tokens_count': len(tokens),
                'reduction_ratio': 1 - (len(processed_text) / max(len(original_text), 1))
            }
        }

def main():
    """Test the enhanced text preprocessor."""
    print("🔧 Testing Enhanced Text Preprocessor")
    print("=" * 50)
    
    preprocessor = TextPreprocessor()
    
    test_texts = [
        "BREAKING!!! You WON'T believe what scientists discovered in your backyard! Click here NOW! 🚨",
        "Reuters reports that the Federal Reserve announced new interest rate policies today.",
        "<p>This is HTML content with <b>bold</b> text and a link to https://example.com</p>",
        "Contact us at john.doe@email.com or call (555) 123-4567 for more information.",
        "This text has... excessive punctuation!!! And multiple    spaces   everywhere.",
        "Short",  # Too short
        "This is a normal news article about economic policies and their impact on society."
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n📰 Test {i}: {text[:50]}...")
        print("-" * 60)
        
        result = preprocessor.preprocess_text(text)
        
        if 'error' not in result:
            print(f"✅ Processed: {result['processed_text'][:100]}...")
            print(f"📊 Stats: {result['processing_stats']}")
            print(f"🎭 Sensational Features: {result['sensational_features']}")
            print(f"🔧 Steps Applied: {', '.join(result['preprocessing_applied'])}")
            
            if result['validation']['warnings']:
                print(f"⚠️  Warnings: {', '.join(result['validation']['warnings'])}")
        else:
            print(f"❌ Error: {result['error']}")
            print(f"❌ Issues: {', '.join(result['validation']['issues'])}")

if __name__ == "__main__":
    main()
