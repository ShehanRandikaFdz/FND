"""
Fallback System for Credibility Analyzer
Provides graceful degradation when models fail or are uncertain.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import time
import logging
from enum import Enum
import re
from collections import defaultdict

class FallbackLevel(Enum):
    """Fallback levels for graceful degradation."""
    FULL_ENSEMBLE = "full_ensemble"
    PARTIAL_ENSEMBLE = "partial_ensemble"
    SINGLE_MODEL = "single_model"
    RULE_BASED = "rule_based"
    HEURISTIC = "heuristic"
    ERROR = "error"

class FallbackSystem:
    """
    Implements graceful degradation strategies for the credibility analyzer.
    """
    
    def __init__(self, analyzer=None):
        self.analyzer = analyzer
        self.logger = self._setup_logging()
        
        # Model availability tracking
        self.model_status = {
            'svm': True,
            'lstm': True,
            'bert': True
        }
        
        # Fallback strategies
        self.fallback_strategies = {
            FallbackLevel.FULL_ENSEMBLE: self._full_ensemble_fallback,
            FallbackLevel.PARTIAL_ENSEMBLE: self._partial_ensemble_fallback,
            FallbackLevel.SINGLE_MODEL: self._single_model_fallback,
            FallbackLevel.RULE_BASED: self._rule_based_fallback,
            FallbackLevel.HEURISTIC: self._heuristic_fallback,
            FallbackLevel.ERROR: self._error_fallback
        }
        
        # Uncertainty thresholds
        self.uncertainty_thresholds = {
            'low_confidence': 0.6,
            'high_uncertainty': 0.4,
            'model_disagreement': 0.3
        }
        
        # Rule-based patterns
        self.credibility_patterns = {
            'high_credibility': [
                r'\b(reuters|ap|associated press|bbc|cnn|nbc|abc|cbs)\b',
                r'\b(according to|study shows|research indicates|expert says)\b',
                r'\b(published in|peer-reviewed|journal|university)\b',
                r'\b(government|official|spokesperson|department)\b'
            ],
            'low_credibility': [
                r'\b(breaking|urgent|shocking|unbelievable|incredible)\b',
                r'\b(you won\'t believe|doctors hate|one weird trick)\b',
                r'\b(click here|must see|must read|what happens next)\b',
                r'\b(conspiracy|cover-up|they don\'t want you to know)\b'
            ]
        }
        
        # Heuristic rules
        self.heuristic_rules = {
            'exclamation_penalty': 0.1,
            'caps_penalty': 0.15,
            'url_bonus': 0.05,
            'length_bonus': 0.02,
            'source_bonus': 0.2
        }
        
        # Performance tracking
        self.fallback_stats = {
            'total_requests': 0,
            'fallback_usage': defaultdict(int),
            'success_rate': 0.0,
            'avg_response_time': 0.0
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for fallback system."""
        logger = logging.getLogger('fallback_system')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def analyze_with_fallback(self, text: str, timeout: float = 10.0) -> Dict[str, any]:
        """Analyze text with fallback mechanisms."""
        start_time = time.time()
        self.fallback_stats['total_requests'] += 1
        
        try:
            # Determine appropriate fallback level
            fallback_level = self._determine_fallback_level(text, timeout)
            
            # Execute fallback strategy
            result = self.fallback_strategies[fallback_level](text)
            
            # Add fallback metadata
            result['fallback_level'] = fallback_level.value
            result['fallback_reason'] = self._get_fallback_reason(fallback_level)
            result['processing_time'] = time.time() - start_time
            
            # Update statistics
            self.fallback_stats['fallback_usage'][fallback_level.value] += 1
            self._update_success_rate(True)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Fallback system error: {e}")
            return self._error_fallback(text, str(e))
    
    def _determine_fallback_level(self, text: str, timeout: float) -> FallbackLevel:
        """Determine appropriate fallback level based on system state."""
        
        # Check model availability
        available_models = sum(1 for status in self.model_status.values() if status)
        
        if available_models == 0:
            return FallbackLevel.HEURISTIC
        
        # Check for timeout constraints
        if timeout < 2.0:
            return FallbackLevel.RULE_BASED
        
        # Check text complexity
        if self._is_simple_text(text):
            return FallbackLevel.RULE_BASED
        
        # Default to full ensemble
        return FallbackLevel.FULL_ENSEMBLE
    
    def _is_simple_text(self, text: str) -> bool:
        """Check if text is simple enough for rule-based analysis."""
        return (
            len(text) < 100 or  # Very short text
            len(text.split()) < 20 or  # Few words
            text.count('!') > 5 or  # Too many exclamations
            text.isupper()  # All caps
        )
    
    def _full_ensemble_fallback(self, text: str) -> Dict[str, any]:
        """Full ensemble prediction with all available models."""
        self.logger.info("Using full ensemble fallback")
        
        if not self.analyzer:
            return self._error_fallback(text, "No analyzer available")
        
        try:
            result = self.analyzer.analyze_credibility(text)
            
            # Check for high uncertainty
            confidence = result.get('confidence', 0.5)
            if confidence < self.uncertainty_thresholds['low_confidence']:
                self.logger.warning(f"Low confidence ({confidence:.3f}), considering partial ensemble")
                return self._partial_ensemble_fallback(text)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Full ensemble failed: {e}")
            return self._partial_ensemble_fallback(text)
    
    def _partial_ensemble_fallback(self, text: str) -> Dict[str, any]:
        """Partial ensemble with available models."""
        self.logger.info("Using partial ensemble fallback")
        
        if not self.analyzer:
            return self._single_model_fallback(text)
        
        try:
            # Try with reduced model set
            available_models = [name for name, status in self.model_status.items() if status]
            
            if len(available_models) >= 2:
                # Use available models
                result = self.analyzer.analyze_credibility(text)
                
                # Add uncertainty warning
                result['uncertainty_warning'] = f"Using {len(available_models)}/{len(self.model_status)} models"
                return result
            else:
                return self._single_model_fallback(text)
                
        except Exception as e:
            self.logger.error(f"Partial ensemble failed: {e}")
            return self._single_model_fallback(text)
    
    def _single_model_fallback(self, text: str) -> Dict[str, any]:
        """Single model prediction (SVM as most reliable)."""
        self.logger.info("Using single model fallback")
        
        if not self.analyzer:
            return self._rule_based_fallback(text)
        
        try:
            # Use SVM as the most reliable single model
            if self.model_status['svm']:
                # This would need to be implemented in the analyzer
                result = self.analyzer._predict_single_model('svm', text)
                
                # Add fallback warning
                result['fallback_warning'] = "Using single model (SVM) prediction"
                result['confidence'] = max(0.3, result.get('confidence', 0.5) - 0.2)
                
                return result
            else:
                return self._rule_based_fallback(text)
                
        except Exception as e:
            self.logger.error(f"Single model failed: {e}")
            return self._rule_based_fallback(text)
    
    def _rule_based_fallback(self, text: str) -> Dict[str, any]:
        """Rule-based credibility assessment."""
        self.logger.info("Using rule-based fallback")
        
        try:
            credibility_score = self._calculate_rule_based_score(text)
            confidence = 0.4  # Lower confidence for rule-based
            
            # Determine label
            label = "Real" if credibility_score > 0.5 else "Fake"
            
            return {
                'label': label,
                'credibility_score': credibility_score,
                'confidence': confidence,
                'fallback_method': 'rule_based',
                'rules_applied': self._get_applied_rules(text)
            }
            
        except Exception as e:
            self.logger.error(f"Rule-based fallback failed: {e}")
            return self._heuristic_fallback(text)
    
    def _heuristic_fallback(self, text: str) -> Dict[str, any]:
        """Heuristic-based credibility assessment."""
        self.logger.info("Using heuristic fallback")
        
        try:
            score = 0.5  # Base score
            
            # Apply heuristic rules
            text_lower = text.lower()
            
            # Exclamation penalty
            exclamation_count = text.count('!')
            score -= exclamation_count * self.heuristic_rules['exclamation_penalty']
            
            # Caps penalty
            caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
            if caps_ratio > 0.3:
                score -= self.heuristic_rules['caps_penalty']
            
            # URL bonus
            if 'http' in text_lower or 'www.' in text_lower:
                score += self.heuristic_rules['url_bonus']
            
            # Length bonus
            if len(text) > 200:
                score += self.heuristic_rules['length_bonus']
            
            # Source bonus
            credible_sources = ['reuters', 'ap', 'bbc', 'cnn', 'nbc']
            if any(source in text_lower for source in credible_sources):
                score += self.heuristic_rules['source_bonus']
            
            # Normalize score
            score = max(0.0, min(1.0, score))
            
            label = "Real" if score > 0.5 else "Fake"
            
            return {
                'label': label,
                'credibility_score': score,
                'confidence': 0.3,  # Low confidence for heuristic
                'fallback_method': 'heuristic',
                'heuristics_applied': self._get_applied_heuristics(text)
            }
            
        except Exception as e:
            self.logger.error(f"Heuristic fallback failed: {e}")
            return self._error_fallback(text, str(e))
    
    def _error_fallback(self, text: str, error: str = "Unknown error") -> Dict[str, any]:
        """Error fallback - return safe default."""
        self.logger.error(f"Error fallback triggered: {error}")
        self._update_success_rate(False)
        
        return {
            'label': 'Unknown',
            'credibility_score': 0.5,
            'confidence': 0.0,
            'error': error,
            'fallback_method': 'error',
            'status': 'error'
        }
    
    def _calculate_rule_based_score(self, text: str) -> float:
        """Calculate credibility score using rule-based patterns."""
        text_lower = text.lower()
        score = 0.5  # Base score
        
        # High credibility patterns
        high_cred_count = 0
        for pattern in self.credibility_patterns['high_credibility']:
            matches = len(re.findall(pattern, text_lower))
            high_cred_count += matches
        
        # Low credibility patterns
        low_cred_count = 0
        for pattern in self.credibility_patterns['low_credibility']:
            matches = len(re.findall(pattern, text_lower))
            low_cred_count += matches
        
        # Adjust score
        score += high_cred_count * 0.1
        score -= low_cred_count * 0.15
        
        # Normalize
        return max(0.0, min(1.0, score))
    
    def _get_applied_rules(self, text: str) -> List[str]:
        """Get list of applied rules."""
        text_lower = text.lower()
        applied_rules = []
        
        for category, patterns in self.credibility_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    applied_rules.append(f"{category}: {pattern}")
        
        return applied_rules
    
    def _get_applied_heuristics(self, text: str) -> List[str]:
        """Get list of applied heuristics."""
        applied_heuristics = []
        
        # Check each heuristic
        if text.count('!') > 0:
            applied_heuristics.append(f"exclamation_penalty: {text.count('!')}")
        
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        if caps_ratio > 0.3:
            applied_heuristics.append(f"caps_penalty: {caps_ratio:.2f}")
        
        if 'http' in text.lower() or 'www.' in text.lower():
            applied_heuristics.append("url_bonus")
        
        if len(text) > 200:
            applied_heuristics.append("length_bonus")
        
        credible_sources = ['reuters', 'ap', 'bbc', 'cnn', 'nbc']
        if any(source in text.lower() for source in credible_sources):
            applied_heuristics.append("source_bonus")
        
        return applied_heuristics
    
    def _get_fallback_reason(self, fallback_level: FallbackLevel) -> str:
        """Get human-readable fallback reason."""
        reasons = {
            FallbackLevel.FULL_ENSEMBLE: "All models available and functioning",
            FallbackLevel.PARTIAL_ENSEMBLE: "Some models unavailable or uncertain",
            FallbackLevel.SINGLE_MODEL: "Limited to single model prediction",
            FallbackLevel.RULE_BASED: "Using rule-based analysis due to constraints",
            FallbackLevel.HEURISTIC: "Using heuristic analysis as last resort",
            FallbackLevel.ERROR: "System error - using safe defaults"
        }
        return reasons.get(fallback_level, "Unknown reason")
    
    def _update_success_rate(self, success: bool) -> None:
        """Update success rate statistics."""
        # Simple moving average
        current_rate = self.fallback_stats['success_rate']
        total_requests = self.fallback_stats['total_requests']
        
        if total_requests == 1:
            self.fallback_stats['success_rate'] = 1.0 if success else 0.0
        else:
            # Exponential moving average
            alpha = 0.1
            self.fallback_stats['success_rate'] = alpha * (1.0 if success else 0.0) + (1 - alpha) * current_rate
    
    def get_system_status(self) -> Dict[str, any]:
        """Get current system status."""
        return {
            'model_status': self.model_status.copy(),
            'fallback_stats': self.fallback_stats.copy(),
            'uncertainty_thresholds': self.uncertainty_thresholds.copy(),
            'available_models': sum(1 for status in self.model_status.values() if status),
            'total_models': len(self.model_status)
        }
    
    def simulate_model_failure(self, model_name: str) -> None:
        """Simulate model failure for testing."""
        if model_name in self.model_status:
            self.model_status[model_name] = False
            self.logger.warning(f"Simulated failure for model: {model_name}")
    
    def restore_model(self, model_name: str) -> None:
        """Restore model after failure."""
        if model_name in self.model_status:
            self.model_status[model_name] = True
            self.logger.info(f"Restored model: {model_name}")

def main():
    """Test the fallback system."""
    print("🛡️ Testing Fallback System")
    print("=" * 50)
    
    # Initialize components
    from credibility_analyzer import CredibilityAnalyzer
    analyzer = CredibilityAnalyzer()
    fallback_system = FallbackSystem(analyzer)
    
    test_texts = [
        "Reuters reports that the Federal Reserve announced new policies.",
        "BREAKING!!! You WON'T believe this!!!",
        "According to experts, this is a significant development.",
        "Click here now for shocking revelations!",
        "The study was published in a peer-reviewed journal.",
        "URGENT: Government officials don't want you to know!"
    ]
    
    print("🧪 Testing normal operation...")
    for i, text in enumerate(test_texts[:3], 1):
        print(f"\nTest {i}: {text[:50]}...")
        result = fallback_system.analyze_with_fallback(text)
        print(f"   Result: {result['label']} (score: {result['credibility_score']:.3f})")
        print(f"   Method: {result['fallback_level']}")
        print(f"   Confidence: {result['confidence']:.3f}")
    
    print("\n🔧 Testing model failure scenarios...")
    
    # Simulate SVM failure
    fallback_system.simulate_model_failure('svm')
    print("\n📉 After SVM failure:")
    result = fallback_system.analyze_with_fallback(test_texts[0])
    print(f"   Result: {result['label']} (score: {result['credibility_score']:.3f})")
    print(f"   Method: {result['fallback_level']}")
    
    # Simulate multiple failures
    fallback_system.simulate_model_failure('lstm')
    fallback_system.simulate_model_failure('bert')
    print("\n📉 After all model failures:")
    result = fallback_system.analyze_with_fallback(test_texts[1])
    print(f"   Result: {result['label']} (score: {result['credibility_score']:.3f})")
    print(f"   Method: {result['fallback_level']}")
    
    # Test with timeout constraint
    print("\n⏱️ Testing with timeout constraint:")
    result = fallback_system.analyze_with_fallback(test_texts[2], timeout=0.5)
    print(f"   Result: {result['label']} (score: {result['credibility_score']:.3f})")
    print(f"   Method: {result['fallback_level']}")
    
    # Restore models
    fallback_system.restore_model('svm')
    fallback_system.restore_model('lstm')
    fallback_system.restore_model('bert')
    
    print(f"\n📊 System Status:")
    status = fallback_system.get_system_status()
    print(f"   Available Models: {status['available_models']}/{status['total_models']}")
    print(f"   Success Rate: {status['fallback_stats']['success_rate']:.1%}")
    print(f"   Total Requests: {status['fallback_stats']['total_requests']}")

if __name__ == "__main__":
    main()
