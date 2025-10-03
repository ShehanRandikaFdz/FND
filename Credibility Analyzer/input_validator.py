"""
Input Validation for Credibility Analyzer
Validates and sanitizes inputs before credibility analysis.
"""

import re
import unicodedata
import html
import urllib.parse
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from enum import Enum

class ValidationLevel(Enum):
    """Validation levels for input sanitization."""
    STRICT = "strict"
    MODERATE = "moderate"
    LENIENT = "lenient"

class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass

@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    sanitized_text: str
    warnings: List[str]
    errors: List[str]
    validation_score: float
    metadata: Dict[str, any]

class InputValidator:
    """
    Validates and sanitizes inputs for the credibility analyzer.
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.MODERATE):
        self.validation_level = validation_level
        self.logger = self._setup_logging()
        
        # Validation thresholds
        self.thresholds = {
            'min_length': 10,
            'max_length': 10000,
            'max_words': 2000,
            'max_lines': 100,
            'max_urls': 10,
            'max_emails': 5,
            'max_special_chars_ratio': 0.3,
            'max_repeated_chars': 3,
            'max_caps_ratio': 0.8
        }
        
        # Security patterns
        self.malicious_patterns = {
            'sql_injection': [
                r'(union|select|insert|update|delete|drop|create|alter)\s+.*',
                r'(\'|\"|;|--|\/\*|\*\/)',
                r'(or|and)\s+.*=.*'
            ],
            'xss_attempts': [
                r'<script[^>]*>.*?</script>',
                r'javascript:',
                r'on\w+\s*=',
                r'<iframe[^>]*>',
                r'<object[^>]*>',
                r'<embed[^>]*>'
            ],
            'path_traversal': [
                r'\.\.\/',
                r'\.\.\\',
                r'%2e%2e%2f',
                r'%2e%2e%5c'
            ],
            'command_injection': [
                r'[;&|`$()]',
                r'exec\s*\(',
                r'eval\s*\(',
                r'system\s*\('
            ]
        }
        
        # Spam patterns
        self.spam_patterns = [
            r'click here',
            r'buy now',
            r'limited time',
            r'act now',
            r'free money',
            r'earn \$?\d+',
            r'work from home',
            r'make money fast',
            r'guaranteed',
            r'no risk'
        ]
        
        # Language detection patterns
        self.language_patterns = {
            'english': r'[a-zA-Z]',
            'numbers': r'[0-9]',
            'special_chars': r'[^\w\s]',
            'unicode': r'[^\x00-\x7F]'
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for input validator."""
        logger = logging.getLogger('input_validator')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def validate_input(self, text: Union[str, Dict[str, str]], 
                      validate_security: bool = True,
                      validate_content: bool = True) -> ValidationResult:
        """
        Validate and sanitize input text.
        
        Args:
            text: Input text or dictionary with 'text' key
            validate_security: Whether to perform security validation
            validate_content: Whether to perform content validation
        
        Returns:
            ValidationResult object with validation results
        """
        # Extract text from input
        if isinstance(text, dict):
            if 'text' not in text:
                raise ValidationError("Input dictionary must contain 'text' key")
            input_text = text['text']
        else:
            input_text = str(text)
        
        # Initialize result
        result = ValidationResult(
            is_valid=True,
            sanitized_text=input_text,
            warnings=[],
            errors=[],
            validation_score=1.0,
            metadata={}
        )
        
        try:
            # Basic validation
            self._validate_basic_properties(input_text, result)
            
            # Security validation
            if validate_security:
                self._validate_security(input_text, result)
            
            # Content validation
            if validate_content:
                self._validate_content(input_text, result)
            
            # Sanitization
            if result.is_valid or self.validation_level == ValidationLevel.LENIENT:
                result.sanitized_text = self._sanitize_text(input_text, result)
            
            # Calculate final validation score
            result.validation_score = self._calculate_validation_score(result)
            
            # Determine final validity
            if result.errors and self.validation_level != ValidationLevel.LENIENT:
                result.is_valid = False
            
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            result.errors.append(f"Validation failed: {str(e)}")
            result.is_valid = False
            result.validation_score = 0.0
        
        return result
    
    def _validate_basic_properties(self, text: str, result: ValidationResult) -> None:
        """Validate basic text properties."""
        
        # Length validation
        if len(text) < self.thresholds['min_length']:
            result.errors.append(f"Text too short (minimum {self.thresholds['min_length']} characters)")
        elif len(text) > self.thresholds['max_length']:
            result.warnings.append(f"Text very long ({len(text)} characters)")
        
        # Word count validation
        word_count = len(text.split())
        if word_count < 3:
            result.errors.append("Text must contain at least 3 words")
        elif word_count > self.thresholds['max_words']:
            result.warnings.append(f"Text has many words ({word_count})")
        
        # Line count validation
        line_count = text.count('\n') + 1
        if line_count > self.thresholds['max_lines']:
            result.warnings.append(f"Text has many lines ({line_count})")
        
        # Store metadata
        result.metadata.update({
            'length': len(text),
            'word_count': word_count,
            'line_count': line_count,
            'char_types': self._analyze_character_types(text)
        })
    
    def _validate_security(self, text: str, result: ValidationResult) -> None:
        """Validate for security threats."""
        
        # Check for malicious patterns
        for threat_type, patterns in self.malicious_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    result.errors.append(f"Potential {threat_type} detected")
                    break
        
        # Check for suspicious URLs
        url_count = len(re.findall(r'https?://[^\s]+', text))
        if url_count > self.thresholds['max_urls']:
            result.warnings.append(f"Too many URLs ({url_count})")
        
        # Check for email addresses
        email_count = len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
        if email_count > self.thresholds['max_emails']:
            result.warnings.append(f"Too many email addresses ({email_count})")
        
        # Check for excessive special characters
        special_char_count = len(re.findall(r'[^\w\s]', text))
        special_char_ratio = special_char_count / max(len(text), 1)
        if special_char_ratio > self.thresholds['max_special_chars_ratio']:
            result.warnings.append(f"High ratio of special characters ({special_char_ratio:.2f})")
        
        # Store security metadata
        result.metadata['security'] = {
            'url_count': url_count,
            'email_count': email_count,
            'special_char_ratio': special_char_ratio,
            'threats_detected': len([e for e in result.errors if 'detected' in e])
        }
    
    def _validate_content(self, text: str, result: ValidationResult) -> None:
        """Validate content quality and characteristics."""
        
        # Check for spam patterns
        spam_matches = 0
        for pattern in self.spam_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                spam_matches += 1
        
        if spam_matches > 2:
            result.warnings.append(f"Potential spam content ({spam_matches} patterns)")
        
        # Check for excessive repetition
        words = text.lower().split()
        if words:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            max_repetition = max(word_counts.values())
            if max_repetition > 5:
                result.warnings.append(f"Excessive word repetition (max: {max_repetition})")
        
        # Check for repeated characters
        for char in set(text):
            if char.isalpha() and text.count(char * 4) > 0:
                result.warnings.append(f"Excessive character repetition: '{char}'")
        
        # Check capitalization
        caps_count = sum(1 for c in text if c.isupper())
        caps_ratio = caps_count / max(len(text), 1)
        if caps_ratio > self.thresholds['max_caps_ratio']:
            result.warnings.append(f"Excessive capitalization ({caps_ratio:.2f})")
        
        # Language analysis
        language_analysis = self._analyze_language(text)
        result.metadata['language'] = language_analysis
        
        if language_analysis['english_ratio'] < 0.5:
            result.warnings.append("Low English content ratio")
    
    def _analyze_character_types(self, text: str) -> Dict[str, int]:
        """Analyze character type distribution."""
        return {
            'letters': sum(1 for c in text if c.isalpha()),
            'digits': sum(1 for c in text if c.isdigit()),
            'spaces': sum(1 for c in text if c.isspace()),
            'punctuation': sum(1 for c in text if c in '.,!?;:'),
            'special': sum(1 for c in text if not c.isalnum() and not c.isspace())
        }
    
    def _analyze_language(self, text: str) -> Dict[str, float]:
        """Analyze language characteristics."""
        total_chars = len(text)
        if total_chars == 0:
            return {'english_ratio': 0.0, 'numeric_ratio': 0.0, 'special_ratio': 0.0}
        
        english_chars = len(re.findall(self.language_patterns['english'], text))
        numeric_chars = len(re.findall(self.language_patterns['numbers'], text))
        special_chars = len(re.findall(self.language_patterns['special_chars'], text))
        
        return {
            'english_ratio': english_chars / total_chars,
            'numeric_ratio': numeric_chars / total_chars,
            'special_ratio': special_chars / total_chars
        }
    
    def _sanitize_text(self, text: str, result: ValidationResult) -> str:
        """Sanitize text based on validation results."""
        sanitized = text
        
        # HTML decoding
        sanitized = html.unescape(sanitized)
        
        # Unicode normalization
        sanitized = unicodedata.normalize('NFKC', sanitized)
        
        # Remove excessive whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        
        # Remove potential script tags (basic)
        sanitized = re.sub(r'<script[^>]*>.*?</script>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove potential iframe tags
        sanitized = re.sub(r'<iframe[^>]*>.*?</iframe>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        
        # URL encoding for suspicious characters
        suspicious_chars = ['<', '>', '"', "'", '&']
        for char in suspicious_chars:
            if char in sanitized:
                sanitized = sanitized.replace(char, urllib.parse.quote(char))
        
        # Log sanitization
        if sanitized != text:
            result.metadata['sanitization_applied'] = True
            result.metadata['original_length'] = len(text)
            result.metadata['sanitized_length'] = len(sanitized)
        
        return sanitized
    
    def _calculate_validation_score(self, result: ValidationResult) -> float:
        """Calculate overall validation score."""
        score = 1.0
        
        # Penalize errors more heavily than warnings
        score -= len(result.errors) * 0.3
        score -= len(result.warnings) * 0.1
        
        # Additional penalties based on content
        if result.metadata.get('security', {}).get('threats_detected', 0) > 0:
            score -= 0.5
        
        # Bonus for good characteristics
        char_types = result.metadata.get('char_types', {})
        if char_types.get('letters', 0) > 50:  # Substantial text content
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def validate_batch(self, texts: List[Union[str, Dict[str, str]]]) -> List[ValidationResult]:
        """Validate a batch of texts."""
        results = []
        
        for i, text in enumerate(texts):
            try:
                result = self.validate_input(text)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Batch validation error for item {i}: {e}")
                # Create error result
                error_result = ValidationResult(
                    is_valid=False,
                    sanitized_text=str(text) if isinstance(text, str) else str(text.get('text', '')),
                    warnings=[],
                    errors=[f"Batch validation failed: {str(e)}"],
                    validation_score=0.0,
                    metadata={'batch_index': i}
                )
                results.append(error_result)
        
        return results
    
    def get_validation_summary(self, results: List[ValidationResult]) -> Dict[str, any]:
        """Get summary of validation results."""
        total = len(results)
        valid_count = sum(1 for r in results if r.is_valid)
        
        total_errors = sum(len(r.errors) for r in results)
        total_warnings = sum(len(r.warnings) for r in results)
        
        avg_score = sum(r.validation_score for r in results) / total if total > 0 else 0.0
        
        return {
            'total_inputs': total,
            'valid_inputs': valid_count,
            'invalid_inputs': total - valid_count,
            'validity_rate': valid_count / total if total > 0 else 0.0,
            'total_errors': total_errors,
            'total_warnings': total_warnings,
            'average_score': avg_score,
            'error_rate': total_errors / total if total > 0 else 0.0
        }

def main():
    """Test the input validator."""
    print("🔍 Testing Input Validator")
    print("=" * 50)
    
    # Initialize validator
    validator = InputValidator(ValidationLevel.MODERATE)
    
    test_inputs = [
        "This is a normal news article about recent developments.",
        "BREAKING!!! You WON'T believe this!!! Click here NOW!!!",
        "<script>alert('xss')</script>This is malicious content.",
        "SELECT * FROM users WHERE id = 1; DROP TABLE users;",
        "This text is too short.",
        "Normal article with some URLs: https://example.com and http://test.org",
        "Text with emails: contact@example.com and support@test.org",
        "This text has excessive CAPITALIZATION and REPETITION of words words words.",
        "Normal text with proper formatting and reasonable length for analysis.",
        {"text": "Text from dictionary input format."}
    ]
    
    print("🧪 Testing individual validation...")
    for i, input_text in enumerate(test_inputs[:5], 1):
        print(f"\nTest {i}: {str(input_text)[:50]}...")
        result = validator.validate_input(input_text)
        
        print(f"   Valid: {result.is_valid}")
        print(f"   Score: {result.validation_score:.3f}")
        print(f"   Errors: {len(result.errors)}")
        print(f"   Warnings: {len(result.warnings)}")
        
        if result.errors:
            print(f"   Error details: {result.errors}")
        if result.warnings:
            print(f"   Warning details: {result.warnings}")
    
    print("\n📦 Testing batch validation...")
    batch_results = validator.validate_batch(test_inputs)
    summary = validator.get_validation_summary(batch_results)
    
    print(f"   Total inputs: {summary['total_inputs']}")
    print(f"   Valid inputs: {summary['valid_inputs']}")
    print(f"   Validity rate: {summary['validity_rate']:.1%}")
    print(f"   Average score: {summary['average_score']:.3f}")
    print(f"   Total errors: {summary['total_errors']}")
    print(f"   Total warnings: {summary['total_warnings']}")
    
    print("\n🔧 Testing different validation levels...")
    
    # Test strict validation
    strict_validator = InputValidator(ValidationLevel.STRICT)
    result = strict_validator.validate_input("Short text")
    print(f"   Strict validation - 'Short text': Valid={result.is_valid}")
    
    # Test lenient validation
    lenient_validator = InputValidator(ValidationLevel.LENIENT)
    result = lenient_validator.validate_input("Short text")
    print(f"   Lenient validation - 'Short text': Valid={result.is_valid}")

if __name__ == "__main__":
    main()
