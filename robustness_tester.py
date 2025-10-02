"""
Robustness Testing for Credibility Analyzer
Tests the system against adversarial inputs and edge cases.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import random
import string
import re
from credibility_analyzer import CredibilityAnalyzer
from explainability_engine import ExplainabilityEngine

class RobustnessTester:
    """
    Tests the credibility analyzer against various adversarial inputs and edge cases.
    """
    
    def __init__(self, analyzer: CredibilityAnalyzer):
        self.analyzer = analyzer
        self.explainer = ExplainabilityEngine(analyzer)
        
        # Base texts for adversarial testing
        self.real_text = "Reuters reports that the Federal Reserve announced new interest rate policies today after careful economic analysis."
        self.fake_text = "BREAKING!!! You WON'T believe what scientists discovered! Doctors HATE this one weird trick! Click here NOW!"
        
        # Adversarial patterns
        self.adversarial_patterns = {
            'typos': ['teh', 'adn', 'recieve', 'seperate', 'definately', 'occured'],
            'homophones': ['there', 'their', 'theyre', 'to', 'too', 'two', 'its', 'its'],
            'repeated_chars': ['reeeeally', 'sooo', 'nooo', 'yesss', 'waaay'],
            'missing_spaces': ['thissentence', 'break ing', 'new s', 'go vernment'],
            'extra_spaces': ['  multiple   spaces  ', '   tabs\t\t\t', '   newlines\n\n\n'],
        }
    
    def test_typo_robustness(self, text: str, num_typos: int = 3) -> Dict[str, any]:
        """Test robustness against typos and misspellings."""
        words = text.split()
        typo_positions = random.sample(range(len(words)), min(num_typos, len(words)))
        
        adversarial_text = text
        for pos in typo_positions:
            original_word = words[pos]
            # Introduce random typo
            if len(original_word) > 2:
                char_pos = random.randint(1, len(original_word) - 2)
                typo_word = original_word[:char_pos] + random.choice(string.ascii_lowercase) + original_word[char_pos + 1:]
                adversarial_text = adversarial_text.replace(original_word, typo_word)
        
        original_result = self.analyzer.analyze_credibility(text)
        adversarial_result = self.analyzer.analyze_credibility(adversarial_text)
        
        return {
            'test_type': 'typo_robustness',
            'original_text': text,
            'adversarial_text': adversarial_text,
            'original_prediction': original_result.get('label', 'Unknown'),
            'adversarial_prediction': adversarial_result.get('label', 'Unknown'),
            'prediction_stable': original_result.get('label') == adversarial_result.get('label'),
            'confidence_change': abs(original_result.get('confidence', 0) - adversarial_result.get('confidence', 0)),
            'score_change': abs(original_result.get('credibility_score', 0) - adversarial_result.get('credibility_score', 0))
        }
    
    def test_capitalization_robustness(self, text: str) -> Dict[str, any]:
        """Test robustness against different capitalization patterns."""
        variations = {
            'all_lower': text.lower(),
            'all_upper': text.upper(),
            'title_case': text.title(),
            'random_caps': ''.join(c.upper() if random.random() < 0.5 else c.lower() for c in text),
            'inverted_caps': ''.join(c.lower() if c.isupper() else c.upper() for c in text)
        }
        
        original_result = self.analyzer.analyze_credibility(text)
        variation_results = {}
        
        for variation_name, variation_text in variations.items():
            result = self.analyzer.analyze_credibility(variation_text)
            variation_results[variation_name] = {
                'prediction': result.get('label', 'Unknown'),
                'confidence': result.get('confidence', 0),
                'score': result.get('credibility_score', 0),
                'stable': result.get('label') == original_result.get('label')
            }
        
        stable_variations = sum(1 for v in variation_results.values() if v['stable'])
        avg_confidence_change = np.mean([
            abs(original_result.get('confidence', 0) - v['confidence']) 
            for v in variation_results.values()
        ])
        
        return {
            'test_type': 'capitalization_robustness',
            'original_text': text,
            'original_prediction': original_result.get('label', 'Unknown'),
            'variations': variation_results,
            'stability_rate': stable_variations / len(variations),
            'avg_confidence_change': avg_confidence_change,
            'most_stable_variation': max(variation_results.items(), key=lambda x: x[1]['stable'])[0]
        }
    
    def test_punctuation_robustness(self, text: str) -> Dict[str, any]:
        """Test robustness against punctuation variations."""
        variations = {
            'no_punctuation': re.sub(r'[^\w\s]', '', text),
            'extra_punctuation': text.replace('.', '...').replace('!', '!!!').replace('?', '???'),
            'minimal_punctuation': re.sub(r'[^\w\s\.]', '', text),
            'symbol_replacement': text.replace('!', '*').replace('?', '#').replace('.', '+')
        }
        
        original_result = self.analyzer.analyze_credibility(text)
        variation_results = {}
        
        for variation_name, variation_text in variations.items():
            result = self.analyzer.analyze_credibility(variation_text)
            variation_results[variation_name] = {
                'prediction': result.get('label', 'Unknown'),
                'confidence': result.get('confidence', 0),
                'score': result.get('credibility_score', 0),
                'stable': result.get('label') == original_result.get('label')
            }
        
        stable_variations = sum(1 for v in variation_results.values() if v['stable'])
        
        return {
            'test_type': 'punctuation_robustness',
            'original_text': text,
            'original_prediction': original_result.get('label', 'Unknown'),
            'variations': variation_results,
            'stability_rate': stable_variations / len(variations)
        }
    
    def test_whitespace_robustness(self, text: str) -> Dict[str, any]:
        """Test robustness against whitespace variations."""
        variations = {
            'no_spaces': text.replace(' ', ''),
            'extra_spaces': text.replace(' ', '   '),
            'tabs': text.replace(' ', '\t'),
            'newlines': text.replace(' ', '\n'),
            'mixed_whitespace': text.replace(' ', ' \t\n ')
        }
        
        original_result = self.analyzer.analyze_credibility(text)
        variation_results = {}
        
        for variation_name, variation_text in variations.items():
            result = self.analyzer.analyze_credibility(variation_text)
            variation_results[variation_name] = {
                'prediction': result.get('label', 'Unknown'),
                'confidence': result.get('confidence', 0),
                'score': result.get('credibility_score', 0),
                'stable': result.get('label') == original_result.get('label')
            }
        
        stable_variations = sum(1 for v in variation_results.values() if v['stable'])
        
        return {
            'test_type': 'whitespace_robustness',
            'original_text': text,
            'original_prediction': original_result.get('label', 'Unknown'),
            'variations': variation_results,
            'stability_rate': stable_variations / len(variations)
        }
    
    def test_length_robustness(self, text: str) -> Dict[str, any]:
        """Test robustness against different text lengths."""
        variations = {
            'truncated_25': text[:len(text)//4],
            'truncated_50': text[:len(text)//2],
            'truncated_75': text[:3*len(text)//4],
            'padded_2x': text + ' ' + text,
            'padded_3x': text + ' ' + text + ' ' + text,
            'repeated_words': ' '.join([text.split()[0]] * 50) + ' ' + text
        }
        
        original_result = self.analyzer.analyze_credibility(text)
        variation_results = {}
        
        for variation_name, variation_text in variations.items():
            result = self.analyzer.analyze_credibility(variation_text)
            variation_results[variation_name] = {
                'prediction': result.get('label', 'Unknown'),
                'confidence': result.get('confidence', 0),
                'score': result.get('credibility_score', 0),
                'stable': result.get('label') == original_result.get('label'),
                'text_length': len(variation_text)
            }
        
        stable_variations = sum(1 for v in variation_results.values() if v['stable'])
        
        return {
            'test_type': 'length_robustness',
            'original_text': text,
            'original_prediction': original_result.get('label', 'Unknown'),
            'variations': variation_results,
            'stability_rate': stable_variations / len(variations)
        }
    
    def test_unicode_robustness(self, text: str) -> Dict[str, any]:
        """Test robustness against Unicode variations."""
        variations = {
            'smart_quotes': text.replace('"', '"').replace("'", "'"),
            'em_dash': text.replace('-', '—'),
            'ellipsis': text.replace('...', '…'),
            'mixed_unicode': text.replace('a', 'а').replace('e', 'е'),  # Cyrillic lookalikes
            'accents': text.replace('a', 'á').replace('e', 'é').replace('i', 'í')
        }
        
        original_result = self.analyzer.analyze_credibility(text)
        variation_results = {}
        
        for variation_name, variation_text in variations.items():
            result = self.analyzer.analyze_credibility(variation_text)
            variation_results[variation_name] = {
                'prediction': result.get('label', 'Unknown'),
                'confidence': result.get('confidence', 0),
                'score': result.get('credibility_score', 0),
                'stable': result.get('label') == original_result.get('label')
            }
        
        stable_variations = sum(1 for v in variation_results.values() if v['stable'])
        
        return {
            'test_type': 'unicode_robustness',
            'original_text': text,
            'original_prediction': original_result.get('label', 'Unknown'),
            'variations': variation_results,
            'stability_rate': stable_variations / len(variations)
        }
    
    def test_adversarial_patterns(self, text: str) -> Dict[str, any]:
        """Test against common adversarial patterns."""
        adversarial_text = text
        
        # Add random typos
        words = adversarial_text.split()
        for i in range(min(3, len(words))):
            if random.random() < 0.3:
                word = words[i]
                if len(word) > 2:
                    char_pos = random.randint(1, len(word) - 2)
                    words[i] = word[:char_pos] + random.choice(string.ascii_lowercase) + word[char_pos + 1:]
        
        adversarial_text = ' '.join(words)
        
        # Add extra punctuation
        if random.random() < 0.5:
            adversarial_text += '!!!'
        
        # Add random capitalization
        if random.random() < 0.3:
            words = adversarial_text.split()
            for i in range(len(words)):
                if random.random() < 0.2:
                    words[i] = words[i].upper()
            adversarial_text = ' '.join(words)
        
        original_result = self.analyzer.analyze_credibility(text)
        adversarial_result = self.analyzer.analyze_credibility(adversarial_text)
        
        return {
            'test_type': 'adversarial_patterns',
            'original_text': text,
            'adversarial_text': adversarial_text,
            'original_prediction': original_result.get('label', 'Unknown'),
            'adversarial_prediction': adversarial_result.get('label', 'Unknown'),
            'prediction_stable': original_result.get('label') == adversarial_result.get('label'),
            'confidence_change': abs(original_result.get('confidence', 0) - adversarial_result.get('confidence', 0)),
            'score_change': abs(original_result.get('credibility_score', 0) - adversarial_result.get('credibility_score', 0))
        }
    
    def test_edge_cases(self) -> Dict[str, any]:
        """Test against various edge cases."""
        edge_cases = {
            'empty_string': '',
            'single_word': 'test',
            'numbers_only': '123 456 789',
            'symbols_only': '!@#$%^&*()',
            'very_long': 'word ' * 1000,
            'unicode_heavy': '🚀🎉🔥💯✨🌟💪🎯🏆',
            'mixed_languages': 'Hello 你好 Hola مرحبا',
            'html_tags': '<b>Bold</b> <i>italic</i> <a href="#">link</a>',
            'urls_only': 'https://example.com http://test.org',
            'email_only': 'test@example.com user@domain.org'
        }
        
        results = {}
        
        for case_name, text in edge_cases.items():
            try:
                result = self.analyzer.analyze_credibility(text)
                results[case_name] = {
                    'text': text[:50] + '...' if len(text) > 50 else text,
                    'prediction': result.get('label', 'Error'),
                    'confidence': result.get('confidence', 0),
                    'score': result.get('credibility_score', 0),
                    'error': result.get('error', None),
                    'handled_gracefully': 'error' not in result
                }
            except Exception as e:
                results[case_name] = {
                    'text': text[:50] + '...' if len(text) > 50 else text,
                    'prediction': 'Error',
                    'confidence': 0,
                    'score': 0,
                    'error': str(e),
                    'handled_gracefully': False
                }
        
        graceful_handling_rate = sum(1 for r in results.values() if r['handled_gracefully']) / len(results)
        
        return {
            'test_type': 'edge_cases',
            'results': results,
            'graceful_handling_rate': graceful_handling_rate,
            'total_edge_cases': len(edge_cases)
        }
    
    def run_comprehensive_robustness_test(self) -> Dict[str, any]:
        """Run all robustness tests on both real and fake texts."""
        print("🛡️ Running Comprehensive Robustness Tests")
        print("=" * 60)
        
        test_results = {}
        
        # Test both real and fake texts
        test_texts = {
            'real_text': self.real_text,
            'fake_text': self.fake_text
        }
        
        for text_type, text in test_texts.items():
            print(f"\n📝 Testing {text_type}: {text[:50]}...")
            print("-" * 50)
            
            text_results = {}
            
            # Run all robustness tests
            tests = [
                self.test_typo_robustness,
                self.test_capitalization_robustness,
                self.test_punctuation_robustness,
                self.test_whitespace_robustness,
                self.test_length_robustness,
                self.test_unicode_robustness,
                self.test_adversarial_patterns
            ]
            
            for test_func in tests:
                try:
                    result = test_func(text)
                    text_results[result['test_type']] = result
                    print(f"✅ {result['test_type']}: {'PASSED' if self._is_test_passed(result) else 'FAILED'}")
                except Exception as e:
                    print(f"❌ {test_func.__name__}: ERROR - {e}")
                    text_results[test_func.__name__] = {'error': str(e)}
            
            test_results[text_type] = text_results
        
        # Run edge case tests
        print(f"\n🔍 Testing Edge Cases")
        print("-" * 50)
        edge_results = self.test_edge_cases()
        test_results['edge_cases'] = edge_results
        print(f"✅ Edge cases: {edge_results['graceful_handling_rate']:.1%} handled gracefully")
        
        # Calculate overall robustness score
        overall_score = self._calculate_robustness_score(test_results)
        
        return {
            'test_results': test_results,
            'overall_robustness_score': overall_score,
            'summary': self._generate_robustness_summary(test_results, overall_score)
        }
    
    def _is_test_passed(self, test_result: Dict[str, any]) -> bool:
        """Determine if a robustness test passed."""
        if test_result['test_type'] == 'typo_robustness':
            return test_result['prediction_stable']
        elif test_result['test_type'] == 'adversarial_patterns':
            return test_result['prediction_stable']
        elif 'stability_rate' in test_result:
            return test_result['stability_rate'] >= 0.7  # 70% stability threshold
        else:
            return True
    
    def _calculate_robustness_score(self, test_results: Dict[str, any]) -> float:
        """Calculate overall robustness score."""
        scores = []
        
        for text_type in ['real_text', 'fake_text']:
            if text_type in test_results:
                text_scores = []
                for test_type, result in test_results[text_type].items():
                    if 'error' not in result:
                        if test_type in ['typo_robustness', 'adversarial_patterns']:
                            text_scores.append(1.0 if result['prediction_stable'] else 0.0)
                        elif 'stability_rate' in result:
                            text_scores.append(result['stability_rate'])
                        else:
                            text_scores.append(1.0)
                
                if text_scores:
                    scores.append(np.mean(text_scores))
        
        # Include edge case handling
        if 'edge_cases' in test_results:
            scores.append(test_results['edge_cases']['graceful_handling_rate'])
        
        return np.mean(scores) if scores else 0.0
    
    def _generate_robustness_summary(self, test_results: Dict[str, any], overall_score: float) -> Dict[str, any]:
        """Generate summary of robustness testing results."""
        total_tests = 0
        passed_tests = 0
        
        for text_type in ['real_text', 'fake_text']:
            if text_type in test_results:
                for test_type, result in test_results[text_type].items():
                    total_tests += 1
                    if 'error' not in result and self._is_test_passed(result):
                        passed_tests += 1
        
        return {
            'overall_score': overall_score,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'pass_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'edge_case_handling': test_results.get('edge_cases', {}).get('graceful_handling_rate', 0),
            'recommendations': self._generate_recommendations(test_results, overall_score)
        }
    
    def _generate_recommendations(self, test_results: Dict[str, any], overall_score: float) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        if overall_score < 0.7:
            recommendations.append("Overall robustness is below acceptable threshold. Consider improving text preprocessing.")
        
        # Check specific failure patterns
        for text_type in ['real_text', 'fake_text']:
            if text_type in test_results:
                for test_type, result in test_results[text_type].items():
                    if 'error' not in result and not self._is_test_passed(result):
                        if 'stability_rate' in result and result['stability_rate'] < 0.5:
                            recommendations.append(f"Improve {test_type} robustness - only {result['stability_rate']:.1%} stable")
        
        if test_results.get('edge_cases', {}).get('graceful_handling_rate', 1) < 0.8:
            recommendations.append("Improve edge case handling - some inputs cause errors")
        
        if not recommendations:
            recommendations.append("Robustness testing passed! System is resilient to adversarial inputs.")
        
        return recommendations

def main():
    """Test the robustness tester."""
    print("🛡️ Testing Robustness Tester")
    print("=" * 50)
    
    # Initialize components
    analyzer = CredibilityAnalyzer()
    tester = RobustnessTester(analyzer)
    
    # Run comprehensive test
    results = tester.run_comprehensive_robustness_test()
    
    # Print summary
    print(f"\n📊 Robustness Test Summary")
    print("=" * 60)
    summary = results['summary']
    print(f"🎯 Overall Robustness Score: {summary['overall_score']:.3f}")
    print(f"📈 Test Pass Rate: {summary['pass_rate']:.1%} ({summary['passed_tests']}/{summary['total_tests']})")
    print(f"🛡️ Edge Case Handling: {summary['edge_case_handling']:.1%}")
    
    print(f"\n💡 Recommendations:")
    for i, rec in enumerate(summary['recommendations'], 1):
        print(f"   {i}. {rec}")

if __name__ == "__main__":
    main()
