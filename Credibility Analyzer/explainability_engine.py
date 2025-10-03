"""
Explainability Engine for Credibility Analysis
Provides interpretable explanations for credibility predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from credibility_analyzer import CredibilityAnalyzer
from feature_extractor import AdvancedFeatureExtractor
from text_preprocessor import TextPreprocessor

class ExplainabilityEngine:
    """
    Generates explanations for credibility analysis predictions.
    """
    
    def __init__(self, analyzer: CredibilityAnalyzer):
        self.analyzer = analyzer
        self.feature_extractor = AdvancedFeatureExtractor()
        self.preprocessor = TextPreprocessor()
        
        # Load SVM components for feature importance
        self.svm_model = None
        self.svm_vectorizer = None
        self._load_svm_components()
        
        # Explanation templates
        self.explanation_templates = {
            'credible': [
                "This text appears credible because it {reasons}.",
                "The analysis suggests this is likely real news due to {reasons}.",
                "Several indicators point to this being credible: {reasons}."
            ],
            'not_credible': [
                "This text shows signs of being fake news because it {reasons}.",
                "The analysis indicates this is likely not credible due to {reasons}.",
                "Multiple red flags suggest this may be fake: {reasons}."
            ],
            'uncertain': [
                "The credibility of this text is uncertain because {reasons}.",
                "Mixed signals make it difficult to determine credibility: {reasons}.",
                "The analysis shows conflicting indicators: {reasons}."
            ]
        }
    
    def _load_svm_components(self):
        """Load SVM model and vectorizer for feature importance analysis."""
        try:
            self.svm_model = joblib.load('models/final_linear_svm.pkl')
            self.svm_vectorizer = joblib.load('models/final_vectorizer.pkl')
        except Exception as e:
            print(f"Warning: Could not load SVM components for explainability: {e}")
    
    def get_svm_feature_importance(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """Get top contributing features from SVM model."""
        if self.svm_model is None or self.svm_vectorizer is None:
            return []
        
        try:
            # Preprocess text
            cleaned_text = self.analyzer.preprocess_text(text)
            
            # Transform text
            text_vector = self.svm_vectorizer.transform([cleaned_text])
            
            # Get feature names and coefficients
            feature_names = self.svm_vectorizer.get_feature_names_out()
            
            # For CalibratedClassifierCV, we need to access the base estimator
            if hasattr(self.svm_model, 'calibrated_classifiers_'):
                # Get the first calibrated classifier's base estimator
                base_estimator = self.svm_model.calibrated_classifiers_[0].estimator
                coefficients = base_estimator.coef_[0]
            else:
                coefficients = self.svm_model.coef_[0]
            
            # Get non-zero features (features present in this text)
            text_features = text_vector.toarray()[0]
            feature_contributions = []
            
            for i, (feature_name, coef, feature_value) in enumerate(zip(feature_names, coefficients, text_features)):
                if feature_value > 0:  # Only consider features present in the text
                    contribution = coef * feature_value
                    feature_contributions.append((feature_name, contribution))
            
            # Sort by absolute contribution
            feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            
            return feature_contributions[:top_n]
            
        except Exception as e:
            print(f"Error getting SVM feature importance: {e}")
            return []
    
    def analyze_meta_features(self, text: str) -> Dict[str, any]:
        """Analyze meta-features and their contribution to credibility."""
        features = self.feature_extractor.extract_all_features(text)
        
        # Define feature importance weights (based on empirical analysis)
        feature_weights = {
            'credible_source_mentions': (0.3, 'positive'),
            'fake_indicator_count': (-0.4, 'negative'),
            'authority_word_count': (0.2, 'positive'),
            'uncertainty_word_count': (-0.1, 'negative'),
            'emotional_word_count': (-0.2, 'negative'),
            'exclamation_count_normalized': (-0.15, 'negative'),
            'all_caps_ratio': (-0.1, 'negative'),
            'flesch_reading_ease': (0.1, 'positive'),
            'lexical_diversity': (0.1, 'positive'),
            'urgency_score': (-0.2, 'negative'),
            'sentiment_compound': (0.05, 'mixed'),
            'url_count': (-0.05, 'negative'),
            'short_url_count': (-0.1, 'negative'),
        }
        
        feature_analysis = {
            'positive_indicators': [],
            'negative_indicators': [],
            'neutral_indicators': [],
            'overall_score': 0.0
        }
        
        for feature_name, (weight, polarity) in feature_weights.items():
            if feature_name in features:
                value = features[feature_name]
                contribution = weight * value
                feature_analysis['overall_score'] += contribution
                
                if abs(contribution) > 0.01:  # Only include significant contributions
                    indicator = {
                        'feature': feature_name,
                        'value': value,
                        'contribution': contribution,
                        'description': self._get_feature_description(feature_name, value)
                    }
                    
                    if contribution > 0:
                        feature_analysis['positive_indicators'].append(indicator)
                    elif contribution < 0:
                        feature_analysis['negative_indicators'].append(indicator)
                    else:
                        feature_analysis['neutral_indicators'].append(indicator)
        
        # Sort by contribution magnitude
        feature_analysis['positive_indicators'].sort(key=lambda x: x['contribution'], reverse=True)
        feature_analysis['negative_indicators'].sort(key=lambda x: abs(x['contribution']), reverse=True)
        
        return feature_analysis
    
    def _get_feature_description(self, feature_name: str, value: float) -> str:
        """Get human-readable description of feature."""
        descriptions = {
            'credible_source_mentions': f"mentions {int(value)} credible news sources",
            'fake_indicator_count': f"contains {int(value)} fake news indicators",
            'authority_word_count': f"includes {int(value)} authority/expertise references",
            'uncertainty_word_count': f"uses {int(value)} uncertainty/hedge words",
            'emotional_word_count': f"contains {int(value)} emotional manipulation words",
            'exclamation_count_normalized': f"has high exclamation mark usage ({value:.3f})",
            'all_caps_ratio': f"uses excessive capitalization ({value:.1%})",
            'flesch_reading_ease': f"has readability score of {value:.1f}",
            'lexical_diversity': f"shows lexical diversity of {value:.3f}",
            'urgency_score': f"uses {int(value)} urgency/breaking news terms",
            'sentiment_compound': f"has sentiment score of {value:.3f}",
            'url_count': f"contains {int(value)} URLs",
            'short_url_count': f"uses {int(value)} shortened URLs",
        }
        
        return descriptions.get(feature_name, f"{feature_name}: {value}")
    
    def generate_explanation(self, text: str, prediction_result: Dict[str, any]) -> Dict[str, any]:
        """Generate comprehensive explanation for the prediction."""
        
        # Get SVM feature importance
        svm_features = self.get_svm_feature_importance(text, top_n=5)
        
        # Analyze meta-features
        meta_analysis = self.analyze_meta_features(text)
        
        # Determine credibility status
        credibility_score = prediction_result.get('credibility_score', 0.5)
        confidence = prediction_result.get('confidence', 0.5)
        
        if credibility_score > 0.7 and confidence > 0.8:
            status = 'credible'
        elif credibility_score < 0.3 and confidence > 0.8:
            status = 'not_credible'
        else:
            status = 'uncertain'
        
        # Build explanation reasons
        reasons = []
        
        # Add positive indicators
        for indicator in meta_analysis['positive_indicators'][:3]:
            reasons.append(indicator['description'])
        
        # Add negative indicators
        for indicator in meta_analysis['negative_indicators'][:3]:
            reasons.append(indicator['description'])
        
        # Add SVM insights
        if svm_features:
            top_svm_feature = svm_features[0]
            if abs(top_svm_feature[1]) > 0.1:
                direction = "supports credibility" if top_svm_feature[1] > 0 else "suggests fake news"
                reasons.append(f"key term '{top_svm_feature[0]}' {direction}")
        
        # Format explanation
        if reasons:
            reason_text = ", ".join(reasons[:4])  # Limit to top 4 reasons
            explanation_template = np.random.choice(self.explanation_templates[status])
            explanation = explanation_template.format(reasons=reason_text)
        else:
            explanation = "The analysis could not identify clear indicators for this prediction."
        
        # Compile explanation result
        explanation_result = {
            'explanation': explanation,
            'credibility_status': status,
            'confidence_level': self._get_confidence_level(confidence),
            'key_indicators': {
                'positive': meta_analysis['positive_indicators'][:3],
                'negative': meta_analysis['negative_indicators'][:3],
                'svm_features': svm_features[:5]
            },
            'reasoning_breakdown': {
                'meta_feature_score': meta_analysis['overall_score'],
                'model_agreement': self._analyze_model_agreement(prediction_result),
                'text_quality': self._assess_text_quality(text)
            }
        }
        
        return explanation_result
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Convert confidence score to human-readable level."""
        if confidence >= 0.9:
            return "Very High"
        elif confidence >= 0.8:
            return "High"
        elif confidence >= 0.6:
            return "Moderate"
        elif confidence >= 0.4:
            return "Low"
        else:
            return "Very Low"
    
    def _analyze_model_agreement(self, prediction_result: Dict[str, any]) -> str:
        """Analyze agreement between different models."""
        individual_preds = prediction_result.get('individual_predictions', {})
        
        if len(individual_preds) < 2:
            return "Single model prediction"
        
        # Check if all models agree on the label
        labels = [pred.get('label', 'Unknown') for pred in individual_preds.values()]
        unique_labels = set(labels)
        
        if len(unique_labels) == 1:
            return f"All {len(individual_preds)} models agree"
        else:
            return f"Models disagree ({len(unique_labels)} different predictions)"
    
    def _assess_text_quality(self, text: str) -> str:
        """Assess overall text quality."""
        validation = self.preprocessor.validate_text(text)
        
        if not validation['is_valid']:
            return "Poor (validation issues)"
        elif validation['warnings']:
            return "Fair (some concerns)"
        else:
            return "Good"
    
    def explain_prediction(self, text: str) -> Dict[str, any]:
        """Main function to explain a credibility prediction."""
        # Get prediction from analyzer
        prediction_result = self.analyzer.analyze_credibility(text)
        
        if 'error' in prediction_result:
            return {
                'error': prediction_result['error'],
                'explanation': "Could not analyze text due to error."
            }
        
        # Generate explanation
        explanation = self.generate_explanation(text, prediction_result)
        
        # Combine prediction and explanation
        result = prediction_result.copy()
        result['explanation_details'] = explanation
        
        return result

def main():
    """Test the explainability engine."""
    print("🔍 Testing Explainability Engine")
    print("=" * 50)
    
    # Initialize components
    analyzer = CredibilityAnalyzer()
    explainer = ExplainabilityEngine(analyzer)
    
    test_texts = [
        "Reuters reports that the Federal Reserve announced new interest rate policies today after careful economic analysis.",
        "BREAKING!!! You WON'T believe what scientists discovered! Doctors HATE this one weird trick! Click here NOW!",
        "According to a study published in Nature, researchers have found new evidence of water on Mars.",
        "URGENT: This shocking discovery will change everything! Government officials don't want you to know the truth!",
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n📰 Test {i}: {text[:60]}...")
        print("-" * 70)
        
        result = explainer.explain_prediction(text)
        
        if 'error' not in result:
            print(f"🎯 Prediction: {result['label']} (score: {result['credibility_score']:.3f})")
            print(f"📊 Confidence: {result['confidence']:.3f}")
            print(f"🏷️  Status: {result['credibility_status']}")
            
            explanation = result['explanation_details']
            print(f"\n💡 Explanation: {explanation['explanation']}")
            print(f"🔍 Confidence Level: {explanation['confidence_level']}")
            print(f"🤝 Model Agreement: {explanation['reasoning_breakdown']['model_agreement']}")
            
            # Show key indicators
            if explanation['key_indicators']['positive']:
                print("\n✅ Positive Indicators:")
                for indicator in explanation['key_indicators']['positive']:
                    print(f"   • {indicator['description']}")
            
            if explanation['key_indicators']['negative']:
                print("\n❌ Negative Indicators:")
                for indicator in explanation['key_indicators']['negative']:
                    print(f"   • {indicator['description']}")
            
            if explanation['key_indicators']['svm_features']:
                print("\n🔤 Key Terms (SVM):")
                for term, contribution in explanation['key_indicators']['svm_features'][:3]:
                    direction = "+" if contribution > 0 else "-"
                    print(f"   • '{term}' ({direction}{abs(contribution):.3f})")
        else:
            print(f"❌ Error: {result['error']}")

if __name__ == "__main__":
    main()
