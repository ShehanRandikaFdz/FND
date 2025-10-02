"""
Bias Detection for Credibility Analyzer
Audits the system for bias across topics, sources, and demographics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import re
from collections import defaultdict, Counter
from credibility_analyzer import CredibilityAnalyzer
from feature_extractor import AdvancedFeatureExtractor
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class BiasDetector:
    """
    Detects and analyzes bias in the credibility analyzer.
    """
    
    def __init__(self, analyzer: CredibilityAnalyzer):
        self.analyzer = analyzer
        self.feature_extractor = AdvancedFeatureExtractor()
        
        # Define bias categories and keywords
        self.bias_categories = {
            'political': {
                'liberal': ['democrat', 'liberal', 'progressive', 'left-wing', 'biden', 'harris'],
                'conservative': ['republican', 'conservative', 'trump', 'pence', 'right-wing', 'gop'],
                'neutral': ['government', 'congress', 'senate', 'house', 'policy', 'legislation']
            },
            'topics': {
                'health': ['health', 'medical', 'doctor', 'hospital', 'disease', 'covid', 'vaccine'],
                'economy': ['economy', 'economic', 'market', 'stock', 'inflation', 'recession'],
                'politics': ['election', 'vote', 'campaign', 'president', 'senator', 'mayor'],
                'technology': ['tech', 'technology', 'ai', 'artificial intelligence', 'software', 'internet'],
                'environment': ['climate', 'environment', 'global warming', 'pollution', 'renewable'],
                'crime': ['crime', 'police', 'arrest', 'investigation', 'criminal', 'justice']
            },
            'sources': {
                'mainstream': ['reuters', 'ap', 'associated press', 'bbc', 'cnn', 'nbc', 'abc', 'cbs'],
                'alternative': ['breitbart', 'infowars', 'daily stormer', 'natural news'],
                'social_media': ['twitter', 'facebook', 'instagram', 'tiktok', 'youtube'],
                'blogs': ['blog', 'opinion', 'commentary', 'editorial']
            },
            'demographics': {
                'gender_terms': ['man', 'woman', 'male', 'female', 'he', 'she', 'his', 'her'],
                'racial_terms': ['black', 'white', 'hispanic', 'asian', 'native', 'immigrant'],
                'age_terms': ['young', 'old', 'elderly', 'teenager', 'millennial', 'boomer'],
                'religious_terms': ['christian', 'muslim', 'jewish', 'hindu', 'buddhist', 'atheist']
            },
            'emotional': {
                'positive': ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic'],
                'negative': ['bad', 'terrible', 'awful', 'horrible', 'disgusting', 'hateful'],
                'neutral': ['average', 'normal', 'standard', 'typical', 'regular', 'ordinary']
            }
        }
        
        # Bias metrics thresholds
        self.bias_thresholds = {
            'statistical_parity': 0.1,  # Max difference in prediction rates
            'equalized_odds': 0.15,     # Max difference in TPR/FPR
            'demographic_parity': 0.1,  # Max difference across demographics
            'topic_bias': 0.2           # Max prediction difference across topics
        }
    
    def detect_topic_bias(self, test_data: List[Dict[str, str]]) -> Dict[str, any]:
        """Detect bias across different topics."""
        print("🔍 Detecting topic bias...")
        
        topic_predictions = defaultdict(list)
        
        for sample in test_data:
            text = sample['text']
            true_label = sample['label']
            
            # Classify text into topics
            topics = self._classify_topics(text)
            
            # Get prediction
            try:
                result = self.analyzer.analyze_credibility(text)
                prediction = result.get('credibility_score', 0.5)
                confidence = result.get('confidence', 0.5)
                
                for topic in topics:
                    topic_predictions[topic].append({
                        'prediction': prediction,
                        'confidence': confidence,
                        'true_label': true_label
                    })
            except Exception as e:
                print(f"Warning: Error processing text: {e}")
                continue
        
        # Analyze bias for each topic
        topic_bias_analysis = {}
        
        for topic, predictions in topic_predictions.items():
            if len(predictions) >= 10:  # Minimum sample size
                pred_scores = [p['prediction'] for p in predictions]
                conf_scores = [p['confidence'] for p in predictions]
                true_labels = [p['true_label'] for p in predictions]
                
                # Calculate bias metrics
                bias_metrics = self._calculate_topic_bias_metrics(pred_scores, conf_scores, true_labels)
                topic_bias_analysis[topic] = bias_metrics
        
        return {
            'topic_bias_analysis': topic_bias_analysis,
            'overall_topic_bias': self._calculate_overall_topic_bias(topic_bias_analysis)
        }
    
    def detect_source_bias(self, test_data: List[Dict[str, str]]) -> Dict[str, any]:
        """Detect bias across different news sources."""
        print("🔍 Detecting source bias...")
        
        source_predictions = defaultdict(list)
        
        for sample in test_data:
            text = sample['text']
            true_label = sample['label']
            
            # Identify source type
            source_type = self._identify_source_type(text)
            
            # Get prediction
            try:
                result = self.analyzer.analyze_credibility(text)
                prediction = result.get('credibility_score', 0.5)
                confidence = result.get('confidence', 0.5)
                
                source_predictions[source_type].append({
                    'prediction': prediction,
                    'confidence': confidence,
                    'true_label': true_label
                })
            except Exception as e:
                print(f"Warning: Error processing text: {e}")
                continue
        
        # Analyze bias for each source type
        source_bias_analysis = {}
        
        for source_type, predictions in source_predictions.items():
            if len(predictions) >= 10:  # Minimum sample size
                pred_scores = [p['prediction'] for p in predictions]
                conf_scores = [p['confidence'] for p in predictions]
                true_labels = [p['true_label'] for p in predictions]
                
                # Calculate bias metrics
                bias_metrics = self._calculate_source_bias_metrics(pred_scores, conf_scores, true_labels)
                source_bias_analysis[source_type] = bias_metrics
        
        return {
            'source_bias_analysis': source_bias_analysis,
            'overall_source_bias': self._calculate_overall_source_bias(source_bias_analysis)
        }
    
    def detect_demographic_bias(self, test_data: List[Dict[str, str]]) -> Dict[str, any]:
        """Detect bias across demographic groups."""
        print("🔍 Detecting demographic bias...")
        
        demographic_predictions = defaultdict(list)
        
        for sample in test_data:
            text = sample['text']
            true_label = sample['label']
            
            # Identify demographic mentions
            demographics = self._identify_demographics(text)
            
            # Get prediction
            try:
                result = self.analyzer.analyze_credibility(text)
                prediction = result.get('credibility_score', 0.5)
                confidence = result.get('confidence', 0.5)
                
                for demo in demographics:
                    demographic_predictions[demo].append({
                        'prediction': prediction,
                        'confidence': confidence,
                        'true_label': true_label
                    })
            except Exception as e:
                print(f"Warning: Error processing text: {e}")
                continue
        
        # Analyze bias for each demographic
        demographic_bias_analysis = {}
        
        for demo, predictions in demographic_predictions.items():
            if len(predictions) >= 10:  # Minimum sample size
                pred_scores = [p['prediction'] for p in predictions]
                conf_scores = [p['confidence'] for p in predictions]
                true_labels = [p['true_label'] for p in predictions]
                
                # Calculate bias metrics
                bias_metrics = self._calculate_demographic_bias_metrics(pred_scores, conf_scores, true_labels)
                demographic_bias_analysis[demo] = bias_metrics
        
        return {
            'demographic_bias_analysis': demographic_bias_analysis,
            'overall_demographic_bias': self._calculate_overall_demographic_bias(demographic_bias_analysis)
        }
    
    def detect_political_bias(self, test_data: List[Dict[str, str]]) -> Dict[str, any]:
        """Detect political bias in predictions."""
        print("🔍 Detecting political bias...")
        
        political_predictions = defaultdict(list)
        
        for sample in test_data:
            text = sample['text']
            true_label = sample['label']
            
            # Identify political orientation
            political_orientation = self._identify_political_orientation(text)
            
            # Get prediction
            try:
                result = self.analyzer.analyze_credibility(text)
                prediction = result.get('credibility_score', 0.5)
                confidence = result.get('confidence', 0.5)
                
                political_predictions[political_orientation].append({
                    'prediction': prediction,
                    'confidence': confidence,
                    'true_label': true_label
                })
            except Exception as e:
                print(f"Warning: Error processing text: {e}")
                continue
        
        # Analyze bias for each political orientation
        political_bias_analysis = {}
        
        for orientation, predictions in political_predictions.items():
            if len(predictions) >= 10:  # Minimum sample size
                pred_scores = [p['prediction'] for p in predictions]
                conf_scores = [p['confidence'] for p in predictions]
                true_labels = [p['true_label'] for p in predictions]
                
                # Calculate bias metrics
                bias_metrics = self._calculate_political_bias_metrics(pred_scores, conf_scores, true_labels)
                political_bias_analysis[orientation] = bias_metrics
        
        return {
            'political_bias_analysis': political_bias_analysis,
            'overall_political_bias': self._calculate_overall_political_bias(political_bias_analysis)
        }
    
    def detect_emotional_bias(self, test_data: List[Dict[str, str]]) -> Dict[str, any]:
        """Detect bias based on emotional content."""
        print("🔍 Detecting emotional bias...")
        
        emotional_predictions = defaultdict(list)
        
        for sample in test_data:
            text = sample['text']
            true_label = sample['label']
            
            # Identify emotional tone
            emotional_tone = self._identify_emotional_tone(text)
            
            # Get prediction
            try:
                result = self.analyzer.analyze_credibility(text)
                prediction = result.get('credibility_score', 0.5)
                confidence = result.get('confidence', 0.5)
                
                emotional_predictions[emotional_tone].append({
                    'prediction': prediction,
                    'confidence': confidence,
                    'true_label': true_label
                })
            except Exception as e:
                print(f"Warning: Error processing text: {e}")
                continue
        
        # Analyze bias for each emotional tone
        emotional_bias_analysis = {}
        
        for tone, predictions in emotional_predictions.items():
            if len(predictions) >= 10:  # Minimum sample size
                pred_scores = [p['prediction'] for p in predictions]
                conf_scores = [p['confidence'] for p in predictions]
                true_labels = [p['true_label'] for p in predictions]
                
                # Calculate bias metrics
                bias_metrics = self._calculate_emotional_bias_metrics(pred_scores, conf_scores, true_labels)
                emotional_bias_analysis[tone] = bias_metrics
        
        return {
            'emotional_bias_analysis': emotional_bias_analysis,
            'overall_emotional_bias': self._calculate_overall_emotional_bias(emotional_bias_analysis)
        }
    
    def _classify_topics(self, text: str) -> List[str]:
        """Classify text into topics."""
        text_lower = text.lower()
        topics = []
        
        for topic, keywords in self.bias_categories['topics'].items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)
        
        return topics if topics else ['general']
    
    def _identify_source_type(self, text: str) -> str:
        """Identify the type of news source."""
        text_lower = text.lower()
        
        for source_type, keywords in self.bias_categories['sources'].items():
            if any(keyword in text_lower for keyword in keywords):
                return source_type
        
        return 'unknown'
    
    def _identify_demographics(self, text: str) -> List[str]:
        """Identify demographic mentions in text."""
        text_lower = text.lower()
        demographics = []
        
        for demo_type, keywords in self.bias_categories['demographics'].items():
            if any(keyword in text_lower for keyword in keywords):
                demographics.append(demo_type)
        
        return demographics if demographics else ['none']
    
    def _identify_political_orientation(self, text: str) -> str:
        """Identify political orientation of text."""
        text_lower = text.lower()
        
        # Count mentions of each political category
        liberal_count = sum(1 for keyword in self.bias_categories['political']['liberal'] if keyword in text_lower)
        conservative_count = sum(1 for keyword in self.bias_categories['political']['conservative'] if keyword in text_lower)
        neutral_count = sum(1 for keyword in self.bias_categories['political']['neutral'] if keyword in text_lower)
        
        if liberal_count > conservative_count and liberal_count > neutral_count:
            return 'liberal'
        elif conservative_count > liberal_count and conservative_count > neutral_count:
            return 'conservative'
        elif neutral_count > 0:
            return 'neutral'
        else:
            return 'non-political'
    
    def _identify_emotional_tone(self, text: str) -> str:
        """Identify emotional tone of text."""
        text_lower = text.lower()
        
        # Count emotional words
        positive_count = sum(1 for keyword in self.bias_categories['emotional']['positive'] if keyword in text_lower)
        negative_count = sum(1 for keyword in self.bias_categories['emotional']['negative'] if keyword in text_lower)
        neutral_count = sum(1 for keyword in self.bias_categories['emotional']['neutral'] if keyword in text_lower)
        
        if positive_count > negative_count and positive_count > neutral_count:
            return 'positive'
        elif negative_count > positive_count and negative_count > neutral_count:
            return 'negative'
        elif neutral_count > 0:
            return 'neutral'
        else:
            return 'mixed'
    
    def _calculate_topic_bias_metrics(self, predictions: List[float], confidences: List[float], 
                                    true_labels: List[int]) -> Dict[str, float]:
        """Calculate bias metrics for topic analysis."""
        predictions = np.array(predictions)
        confidences = np.array(confidences)
        true_labels = np.array(true_labels)
        
        return {
            'avg_prediction': np.mean(predictions),
            'avg_confidence': np.mean(confidences),
            'prediction_std': np.std(predictions),
            'confidence_std': np.std(confidences),
            'accuracy': np.mean((predictions > 0.5) == true_labels),
            'sample_size': len(predictions)
        }
    
    def _calculate_source_bias_metrics(self, predictions: List[float], confidences: List[float], 
                                     true_labels: List[int]) -> Dict[str, float]:
        """Calculate bias metrics for source analysis."""
        return self._calculate_topic_bias_metrics(predictions, confidences, true_labels)
    
    def _calculate_demographic_bias_metrics(self, predictions: List[float], confidences: List[float], 
                                          true_labels: List[int]) -> Dict[str, float]:
        """Calculate bias metrics for demographic analysis."""
        return self._calculate_topic_bias_metrics(predictions, confidences, true_labels)
    
    def _calculate_political_bias_metrics(self, predictions: List[float], confidences: List[float], 
                                        true_labels: List[int]) -> Dict[str, float]:
        """Calculate bias metrics for political analysis."""
        return self._calculate_topic_bias_metrics(predictions, confidences, true_labels)
    
    def _calculate_emotional_bias_metrics(self, predictions: List[float], confidences: List[float], 
                                        true_labels: List[int]) -> Dict[str, float]:
        """Calculate bias metrics for emotional analysis."""
        return self._calculate_topic_bias_metrics(predictions, confidences, true_labels)
    
    def _calculate_overall_topic_bias(self, topic_analysis: Dict[str, Dict[str, float]]) -> float:
        """Calculate overall topic bias score."""
        if not topic_analysis:
            return 0.0
        
        predictions = [metrics['avg_prediction'] for metrics in topic_analysis.values()]
        return np.std(predictions) if len(predictions) > 1 else 0.0
    
    def _calculate_overall_source_bias(self, source_analysis: Dict[str, Dict[str, float]]) -> float:
        """Calculate overall source bias score."""
        if not source_analysis:
            return 0.0
        
        predictions = [metrics['avg_prediction'] for metrics in source_analysis.values()]
        return np.std(predictions) if len(predictions) > 1 else 0.0
    
    def _calculate_overall_demographic_bias(self, demo_analysis: Dict[str, Dict[str, float]]) -> float:
        """Calculate overall demographic bias score."""
        if not demo_analysis:
            return 0.0
        
        predictions = [metrics['avg_prediction'] for metrics in demo_analysis.values()]
        return np.std(predictions) if len(predictions) > 1 else 0.0
    
    def _calculate_overall_political_bias(self, political_analysis: Dict[str, Dict[str, float]]) -> float:
        """Calculate overall political bias score."""
        if not political_analysis:
            return 0.0
        
        predictions = [metrics['avg_prediction'] for metrics in political_analysis.values()]
        return np.std(predictions) if len(predictions) > 1 else 0.0
    
    def _calculate_overall_emotional_bias(self, emotional_analysis: Dict[str, Dict[str, float]]) -> float:
        """Calculate overall emotional bias score."""
        if not emotional_analysis:
            return 0.0
        
        predictions = [metrics['avg_prediction'] for metrics in emotional_analysis.values()]
        return np.std(predictions) if len(predictions) > 1 else 0.0
    
    def run_comprehensive_bias_audit(self, test_data: List[Dict[str, str]]) -> Dict[str, any]:
        """Run comprehensive bias audit."""
        print("🔍 Running Comprehensive Bias Audit")
        print("=" * 60)
        
        bias_results = {}
        
        # Run all bias detection tests
        bias_results['topic_bias'] = self.detect_topic_bias(test_data)
        bias_results['source_bias'] = self.detect_source_bias(test_data)
        bias_results['demographic_bias'] = self.detect_demographic_bias(test_data)
        bias_results['political_bias'] = self.detect_political_bias(test_data)
        bias_results['emotional_bias'] = self.detect_emotional_bias(test_data)
        
        # Calculate overall bias score
        overall_bias = self._calculate_overall_bias_score(bias_results)
        
        # Generate bias report
        bias_report = self._generate_bias_report(bias_results, overall_bias)
        
        return {
            'bias_results': bias_results,
            'overall_bias_score': overall_bias,
            'bias_report': bias_report
        }
    
    def _calculate_overall_bias_score(self, bias_results: Dict[str, any]) -> float:
        """Calculate overall bias score."""
        bias_scores = []
        
        for bias_type, results in bias_results.items():
            overall_key = f'overall_{bias_type}'
            if overall_key in results:
                bias_scores.append(results[overall_key])
        
        return np.mean(bias_scores) if bias_scores else 0.0
    
    def _generate_bias_report(self, bias_results: Dict[str, any], overall_bias: float) -> Dict[str, any]:
        """Generate comprehensive bias report."""
        report = {
            'overall_bias_score': overall_bias,
            'bias_level': self._categorize_bias_level(overall_bias),
            'recommendations': [],
            'detailed_findings': {}
        }
        
        # Analyze each bias type
        for bias_type, results in bias_results.items():
            report['detailed_findings'][bias_type] = self._analyze_bias_type(bias_type, results)
        
        # Generate recommendations
        report['recommendations'] = self._generate_bias_recommendations(bias_results, overall_bias)
        
        return report
    
    def _categorize_bias_level(self, bias_score: float) -> str:
        """Categorize bias level based on score."""
        if bias_score < 0.05:
            return 'Low'
        elif bias_score < 0.1:
            return 'Moderate'
        elif bias_score < 0.2:
            return 'High'
        else:
            return 'Very High'
    
    def _analyze_bias_type(self, bias_type: str, results: Dict[str, any]) -> Dict[str, any]:
        """Analyze specific bias type results."""
        analysis_key = f'{bias_type}_analysis'
        overall_key = f'overall_{bias_type}'
        
        if analysis_key in results and overall_key in results:
            analysis = results[analysis_key]
            overall_score = results[overall_key]
            
            return {
                'bias_score': overall_score,
                'bias_level': self._categorize_bias_level(overall_score),
                'categories_analyzed': len(analysis),
                'significant_biases': [
                    category for category, metrics in analysis.items()
                    if abs(metrics['avg_prediction'] - 0.5) > 0.1
                ]
            }
        
        return {'error': 'Could not analyze bias type'}
    
    def _generate_bias_recommendations(self, bias_results: Dict[str, any], overall_bias: float) -> List[str]:
        """Generate recommendations based on bias analysis."""
        recommendations = []
        
        if overall_bias > 0.1:
            recommendations.append("Overall bias is high. Consider retraining models with more balanced data.")
        
        # Check specific bias types
        for bias_type, results in bias_results.items():
            overall_key = f'overall_{bias_type}'
            if overall_key in results and results[overall_key] > 0.1:
                recommendations.append(f"High {bias_type.replace('_', ' ')} detected. Review training data for balance.")
        
        if not recommendations:
            recommendations.append("Bias audit passed! System shows minimal bias across tested dimensions.")
        
        return recommendations

def create_sample_test_data() -> List[Dict[str, str]]:
    """Create sample test data for bias detection."""
    sample_data = [
        {'text': 'President Biden announced new healthcare policies today.', 'label': 1},
        {'text': 'Trump supporters rally outside the White House.', 'label': 0},
        {'text': 'Scientists discover new treatment for COVID-19.', 'label': 1},
        {'text': 'BREAKING: Doctors hate this one weird trick!', 'label': 0},
        {'text': 'The economy shows signs of recovery this quarter.', 'label': 1},
        {'text': 'URGENT: Stock market crash imminent!', 'label': 0},
        {'text': 'New climate change legislation passes Congress.', 'label': 1},
        {'text': 'Climate change is a hoax, says expert.', 'label': 0},
        {'text': 'Technology companies report record profits.', 'label': 1},
        {'text': 'Tech giants control everything you see online!', 'label': 0},
        {'text': 'Police arrest suspect in downtown incident.', 'label': 1},
        {'text': 'Police brutality continues across the country!', 'label': 0},
        {'text': 'Young people are driving economic growth.', 'label': 1},
        {'text': 'Millennials are destroying the economy!', 'label': 0},
        {'text': 'Women leaders excel in crisis management.', 'label': 1},
        {'text': 'Men are naturally better leaders than women.', 'label': 0},
        {'text': 'Immigration policies need comprehensive reform.', 'label': 1},
        {'text': 'Immigrants are taking all our jobs!', 'label': 0},
        {'text': 'Religious communities support local charities.', 'label': 1},
        {'text': 'Religious people are brainwashed and naive.', 'label': 0}
    ]
    
    return sample_data

def main():
    """Test the bias detector."""
    print("🔍 Testing Bias Detector")
    print("=" * 50)
    
    # Initialize components
    analyzer = CredibilityAnalyzer()
    bias_detector = BiasDetector(analyzer)
    
    # Create sample test data
    test_data = create_sample_test_data()
    
    # Run comprehensive bias audit
    results = bias_detector.run_comprehensive_bias_audit(test_data)
    
    # Print summary
    print(f"\n📊 Bias Audit Summary")
    print("=" * 60)
    
    bias_report = results['bias_report']
    print(f"🎯 Overall Bias Score: {bias_report['overall_bias_score']:.3f}")
    print(f"🏷️ Bias Level: {bias_report['bias_level']}")
    
    print(f"\n📈 Detailed Findings:")
    for bias_type, findings in bias_report['detailed_findings'].items():
        if 'error' not in findings:
            print(f"   {bias_type.replace('_', ' ').title()}: {findings['bias_level']} ({findings['bias_score']:.3f})")
            if findings['significant_biases']:
                print(f"     Significant biases: {', '.join(findings['significant_biases'])}")
    
    print(f"\n💡 Recommendations:")
    for i, rec in enumerate(bias_report['recommendations'], 1):
        print(f"   {i}. {rec}")

if __name__ == "__main__":
    main()
