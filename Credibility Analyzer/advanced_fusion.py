"""
Advanced Multi-Model Fusion for Credibility Analyzer
Implements sophisticated fusion techniques beyond simple voting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class FusionMethod(Enum):
    """Advanced fusion methods."""
    WEIGHTED_AVERAGE = "weighted_average"
    DYNAMIC_WEIGHTING = "dynamic_weighting"
    STACKING = "stacking"
    BAYESIAN_FUSION = "bayesian_fusion"
    ADAPTIVE_FUSION = "adaptive_fusion"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    META_LEARNING = "meta_learning"

@dataclass
class ModelPrediction:
    """Model prediction data structure."""
    model_name: str
    prediction: float
    confidence: float
    features: Dict[str, float]
    metadata: Dict[str, Any]

@dataclass
class FusionResult:
    """Fusion result data structure."""
    final_prediction: float
    final_confidence: float
    fusion_method: str
    model_weights: Dict[str, float]
    uncertainty: float
    agreement_score: float

class AdvancedFusion:
    """
    Advanced multi-model fusion techniques.
    """
    
    def __init__(self, fusion_method: FusionMethod = FusionMethod.DYNAMIC_WEIGHTING):
        self.fusion_method = fusion_method
        self.meta_model = None
        self.model_performance_history = {}
        self.feature_importance = {}
        
        # Fusion parameters
        self.fusion_params = {
            'min_confidence': 0.3,
            'max_disagreement': 0.4,
            'adaptation_rate': 0.1,
            'history_window': 100
        }
        
        # Model-specific weights (dynamic)
        self.dynamic_weights = {
            'svm': 0.33,
            'lstm': 0.33,
            'bert': 0.34
        }
        
        # Performance tracking
        self.performance_tracker = {
            'svm': {'accuracy': 0.996, 'confidence': 0.95},
            'lstm': {'accuracy': 0.989, 'confidence': 0.90},
            'bert': {'accuracy': 0.975, 'confidence': 0.88}
        }
    
    def fuse_predictions(self, predictions: List[ModelPrediction], 
                        text_features: Dict[str, float] = None) -> FusionResult:
        """
        Fuse multiple model predictions using advanced techniques.
        
        Args:
            predictions: List of model predictions
            text_features: Additional text features for fusion
        
        Returns:
            FusionResult with fused prediction
        """
        if len(predictions) < 2:
            raise ValueError("Need at least 2 predictions for fusion")
        
        # Apply fusion method
        if self.fusion_method == FusionMethod.WEIGHTED_AVERAGE:
            return self._weighted_average_fusion(predictions)
        elif self.fusion_method == FusionMethod.DYNAMIC_WEIGHTING:
            return self._dynamic_weighting_fusion(predictions, text_features)
        elif self.fusion_method == FusionMethod.STACKING:
            return self._stacking_fusion(predictions, text_features)
        elif self.fusion_method == FusionMethod.BAYESIAN_FUSION:
            return self._bayesian_fusion(predictions)
        elif self.fusion_method == FusionMethod.ADAPTIVE_FUSION:
            return self._adaptive_fusion(predictions, text_features)
        elif self.fusion_method == FusionMethod.CONFIDENCE_WEIGHTED:
            return self._confidence_weighted_fusion(predictions)
        elif self.fusion_method == FusionMethod.META_LEARNING:
            return self._meta_learning_fusion(predictions, text_features)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
    
    def _weighted_average_fusion(self, predictions: List[ModelPrediction]) -> FusionResult:
        """Simple weighted average fusion."""
        weights = self._get_model_weights([p.model_name for p in predictions])
        
        weighted_sum = sum(p.prediction * weights[p.model_name] for p in predictions)
        final_prediction = weighted_sum / sum(weights.values())
        
        # Calculate weighted confidence
        weighted_conf = sum(p.confidence * weights[p.model_name] for p in predictions)
        final_confidence = weighted_conf / sum(weights.values())
        
        # Calculate agreement score
        agreement_score = self._calculate_agreement(predictions)
        
        # Calculate uncertainty
        uncertainty = self._calculate_uncertainty(predictions)
        
        return FusionResult(
            final_prediction=final_prediction,
            final_confidence=final_confidence,
            fusion_method="weighted_average",
            model_weights=weights,
            uncertainty=uncertainty,
            agreement_score=agreement_score
        )
    
    def _dynamic_weighting_fusion(self, predictions: List[ModelPrediction], 
                                text_features: Dict[str, float] = None) -> FusionResult:
        """Dynamic weighting based on context and performance."""
        # Base weights from performance
        base_weights = self._get_model_weights([p.model_name for p in predictions])
        
        # Adjust weights based on confidence
        confidence_weights = {}
        for pred in predictions:
            conf_factor = pred.confidence / max([p.confidence for p in predictions])
            confidence_weights[pred.model_name] = conf_factor
        
        # Adjust weights based on text features if available
        feature_weights = {}
        if text_features:
            feature_weights = self._calculate_feature_based_weights(
                [p.model_name for p in predictions], text_features
            )
        
        # Combine weights
        final_weights = {}
        for model_name in [p.model_name for p in predictions]:
            weight = base_weights[model_name]
            weight *= confidence_weights[model_name]
            if model_name in feature_weights:
                weight *= feature_weights[model_name]
            final_weights[model_name] = weight
        
        # Normalize weights
        total_weight = sum(final_weights.values())
        final_weights = {k: v/total_weight for k, v in final_weights.items()}
        
        # Calculate final prediction
        weighted_sum = sum(p.prediction * final_weights[p.model_name] for p in predictions)
        final_prediction = weighted_sum
        
        # Calculate final confidence
        weighted_conf = sum(p.confidence * final_weights[p.model_name] for p in predictions)
        final_confidence = weighted_conf
        
        # Calculate metrics
        agreement_score = self._calculate_agreement(predictions)
        uncertainty = self._calculate_uncertainty(predictions)
        
        return FusionResult(
            final_prediction=final_prediction,
            final_confidence=final_confidence,
            fusion_method="dynamic_weighting",
            model_weights=final_weights,
            uncertainty=uncertainty,
            agreement_score=agreement_score
        )
    
    def _stacking_fusion(self, predictions: List[ModelPrediction], 
                        text_features: Dict[str, float] = None) -> FusionResult:
        """Stacking-based fusion with meta-learning."""
        # Prepare features for meta-model
        fusion_features = self._prepare_stacking_features(predictions, text_features)
        
        # Use meta-model if available, otherwise fallback to weighted average
        if self.meta_model is not None:
            try:
                final_prediction = self.meta_model.predict([fusion_features])[0]
                final_confidence = self._calculate_stacking_confidence(predictions, fusion_features)
            except Exception:
                # Fallback to weighted average
                return self._weighted_average_fusion(predictions)
        else:
            # Fallback to weighted average
            return self._weighted_average_fusion(predictions)
        
        # Calculate weights (for interpretability)
        weights = self._get_model_weights([p.model_name for p in predictions])
        
        agreement_score = self._calculate_agreement(predictions)
        uncertainty = self._calculate_uncertainty(predictions)
        
        return FusionResult(
            final_prediction=final_prediction,
            final_confidence=final_confidence,
            fusion_method="stacking",
            model_weights=weights,
            uncertainty=uncertainty,
            agreement_score=agreement_score
        )
    
    def _bayesian_fusion(self, predictions: List[ModelPrediction]) -> FusionResult:
        """Bayesian fusion considering model uncertainty."""
        # Convert predictions to probability distributions
        model_distributions = []
        model_weights = []
        
        for pred in predictions:
            # Assume Gaussian distribution with confidence as precision
            mean = pred.prediction
            precision = pred.confidence * 10  # Scale confidence to precision
            
            model_distributions.append((mean, precision))
            model_weights.append(self.performance_tracker[pred.model_name]['accuracy'])
        
        # Normalize weights
        total_weight = sum(model_weights)
        model_weights = [w/total_weight for w in model_weights]
        
        # Bayesian fusion
        fused_precision = sum(w * precision for w, (_, precision) in zip(model_weights, model_distributions))
        fused_mean = sum(w * mean * precision for w, (mean, precision) in zip(model_weights, model_distributions)) / fused_precision
        
        final_prediction = fused_mean
        final_confidence = min(1.0, fused_precision / 10)  # Convert back to confidence scale
        
        agreement_score = self._calculate_agreement(predictions)
        uncertainty = self._calculate_uncertainty(predictions)
        
        weights_dict = {p.model_name: w for p, w in zip(predictions, model_weights)}
        
        return FusionResult(
            final_prediction=final_prediction,
            final_confidence=final_confidence,
            fusion_method="bayesian_fusion",
            model_weights=weights_dict,
            uncertainty=uncertainty,
            agreement_score=agreement_score
        )
    
    def _adaptive_fusion(self, predictions: List[ModelPrediction], 
                        text_features: Dict[str, float] = None) -> FusionResult:
        """Adaptive fusion that learns from context."""
        # Analyze prediction patterns
        pred_values = [p.prediction for p in predictions]
        conf_values = [p.confidence for p in predictions]
        
        # Detect prediction patterns
        variance = np.var(pred_values)
        mean_confidence = np.mean(conf_values)
        
        # Adaptive strategy selection
        if variance < 0.1 and mean_confidence > 0.8:
            # High agreement, high confidence - use simple average
            strategy = "average"
        elif variance > 0.3:
            # High disagreement - use confidence weighting
            strategy = "confidence_weighted"
        else:
            # Moderate disagreement - use dynamic weighting
            strategy = "dynamic"
        
        # Apply selected strategy
        if strategy == "average":
            weights = {p.model_name: 1.0/len(predictions) for p in predictions}
        elif strategy == "confidence_weighted":
            weights = self._confidence_weighted_weights(predictions)
        else:  # dynamic
            weights = self._dynamic_weighting_fusion(predictions, text_features).model_weights
        
        # Calculate final prediction
        weighted_sum = sum(p.prediction * weights[p.model_name] for p in predictions)
        final_prediction = weighted_sum
        
        # Adaptive confidence calculation
        if strategy == "average":
            final_confidence = np.mean(conf_values)
        else:
            final_confidence = sum(p.confidence * weights[p.model_name] for p in predictions)
        
        agreement_score = self._calculate_agreement(predictions)
        uncertainty = self._calculate_uncertainty(predictions)
        
        return FusionResult(
            final_prediction=final_prediction,
            final_confidence=final_confidence,
            fusion_method=f"adaptive_{strategy}",
            model_weights=weights,
            uncertainty=uncertainty,
            agreement_score=agreement_score
        )
    
    def _confidence_weighted_fusion(self, predictions: List[ModelPrediction]) -> FusionResult:
        """Confidence-weighted fusion."""
        weights = self._confidence_weighted_weights(predictions)
        
        weighted_sum = sum(p.prediction * weights[p.model_name] for p in predictions)
        final_prediction = weighted_sum
        
        final_confidence = sum(p.confidence * weights[p.model_name] for p in predictions)
        
        agreement_score = self._calculate_agreement(predictions)
        uncertainty = self._calculate_uncertainty(predictions)
        
        return FusionResult(
            final_prediction=final_prediction,
            final_confidence=final_confidence,
            fusion_method="confidence_weighted",
            model_weights=weights,
            uncertainty=uncertainty,
            agreement_score=agreement_score
        )
    
    def _meta_learning_fusion(self, predictions: List[ModelPrediction], 
                            text_features: Dict[str, float] = None) -> FusionResult:
        """Meta-learning based fusion."""
        # This would require training data and is simplified here
        # In practice, you'd train a meta-model on historical fusion performance
        
        # For now, use a combination of dynamic weighting and confidence weighting
        dynamic_result = self._dynamic_weighting_fusion(predictions, text_features)
        confidence_result = self._confidence_weighted_fusion(predictions)
        
        # Combine results with learned weights (simplified)
        alpha = 0.6  # Weight for dynamic vs confidence-based
        final_prediction = alpha * dynamic_result.final_prediction + (1 - alpha) * confidence_result.final_prediction
        final_confidence = alpha * dynamic_result.final_confidence + (1 - alpha) * confidence_result.final_confidence
        
        # Combine model weights
        combined_weights = {}
        for model_name in dynamic_result.model_weights.keys():
            combined_weights[model_name] = alpha * dynamic_result.model_weights[model_name] + (1 - alpha) * confidence_result.model_weights[model_name]
        
        agreement_score = self._calculate_agreement(predictions)
        uncertainty = self._calculate_uncertainty(predictions)
        
        return FusionResult(
            final_prediction=final_prediction,
            final_confidence=final_confidence,
            fusion_method="meta_learning",
            model_weights=combined_weights,
            uncertainty=uncertainty,
            agreement_score=agreement_score
        )
    
    def _get_model_weights(self, model_names: List[str]) -> Dict[str, float]:
        """Get model weights based on performance."""
        weights = {}
        total_performance = 0
        
        for model_name in model_names:
            if model_name in self.performance_tracker:
                performance = self.performance_tracker[model_name]['accuracy']
                weights[model_name] = performance
                total_performance += performance
            else:
                weights[model_name] = 0.5  # Default weight
                total_performance += 0.5
        
        # Normalize weights
        if total_performance > 0:
            weights = {k: v/total_performance for k, v in weights.items()}
        
        return weights
    
    def _confidence_weighted_weights(self, predictions: List[ModelPrediction]) -> Dict[str, float]:
        """Calculate confidence-weighted model weights."""
        weights = {}
        total_confidence = sum(p.confidence for p in predictions)
        
        for pred in predictions:
            weights[pred.model_name] = pred.confidence / total_confidence if total_confidence > 0 else 1.0/len(predictions)
        
        return weights
    
    def _calculate_feature_based_weights(self, model_names: List[str], 
                                       text_features: Dict[str, float]) -> Dict[str, float]:
        """Calculate weights based on text features."""
        # Simplified feature-based weighting
        weights = {}
        
        # Example: Give more weight to BERT for longer texts, SVM for shorter texts
        text_length = text_features.get('char_count', 0)
        
        for model_name in model_names:
            if model_name == 'bert' and text_length > 200:
                weights[model_name] = 1.2
            elif model_name == 'svm' and text_length < 100:
                weights[model_name] = 1.1
            else:
                weights[model_name] = 1.0
        
        return weights
    
    def _prepare_stacking_features(self, predictions: List[ModelPrediction], 
                                 text_features: Dict[str, float] = None) -> List[float]:
        """Prepare features for stacking meta-model."""
        features = []
        
        # Add model predictions
        for pred in predictions:
            features.extend([pred.prediction, pred.confidence])
        
        # Add prediction statistics
        pred_values = [p.prediction for p in predictions]
        conf_values = [p.confidence for p in predictions]
        
        features.extend([
            np.mean(pred_values),
            np.std(pred_values),
            np.mean(conf_values),
            np.std(conf_values),
            self._calculate_agreement(predictions)
        ])
        
        # Add text features if available
        if text_features:
            key_features = ['char_count', 'word_count', 'sentiment_compound', 'flesch_reading_ease']
            for feature in key_features:
                features.append(text_features.get(feature, 0.0))
        
        return features
    
    def _calculate_stacking_confidence(self, predictions: List[ModelPrediction], 
                                     fusion_features: List[float]) -> float:
        """Calculate confidence for stacking fusion."""
        # Simplified confidence calculation
        base_confidence = np.mean([p.confidence for p in predictions])
        agreement_bonus = self._calculate_agreement(predictions) * 0.1
        
        return min(1.0, base_confidence + agreement_bonus)
    
    def _calculate_agreement(self, predictions: List[ModelPrediction]) -> float:
        """Calculate agreement score between predictions."""
        if len(predictions) < 2:
            return 1.0
        
        pred_values = [p.prediction for p in predictions]
        variance = np.var(pred_values)
        
        # Convert variance to agreement score (0-1)
        agreement = max(0.0, 1.0 - variance * 4)  # Scale factor for variance
        return agreement
    
    def _calculate_uncertainty(self, predictions: List[ModelPrediction]) -> float:
        """Calculate uncertainty in fusion."""
        # Uncertainty based on disagreement and confidence
        agreement = self._calculate_agreement(predictions)
        mean_confidence = np.mean([p.confidence for p in predictions])
        
        # Higher disagreement and lower confidence = higher uncertainty
        uncertainty = (1.0 - agreement) * (1.0 - mean_confidence)
        return min(1.0, uncertainty)
    
    def train_meta_model(self, training_data: List[Tuple[List[ModelPrediction], float]]) -> None:
        """Train meta-model for stacking fusion."""
        if not training_data:
            return
        
        # Prepare training features and targets
        X = []
        y = []
        
        for predictions, true_label in training_data:
            features = self._prepare_stacking_features(predictions)
            X.append(features)
            y.append(true_label)
        
        # Train meta-model
        self.meta_model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.meta_model.fit(X, y)
    
    def update_model_performance(self, model_name: str, accuracy: float, confidence: float) -> None:
        """Update model performance for dynamic weighting."""
        if model_name in self.performance_tracker:
            # Exponential moving average
            alpha = 0.1
            self.performance_tracker[model_name]['accuracy'] = (
                alpha * accuracy + (1 - alpha) * self.performance_tracker[model_name]['accuracy']
            )
            self.performance_tracker[model_name]['confidence'] = (
                alpha * confidence + (1 - alpha) * self.performance_tracker[model_name]['confidence']
            )

def main():
    """Test the advanced fusion system."""
    print("🔄 Testing Advanced Fusion System")
    print("=" * 50)
    
    # Initialize fusion system
    fusion = AdvancedFusion(FusionMethod.DYNAMIC_WEIGHTING)
    
    # Create sample predictions
    predictions = [
        ModelPrediction(
            model_name='svm',
            prediction=0.8,
            confidence=0.95,
            features={'tfidf_score': 0.8},
            metadata={'model_version': '1.0'}
        ),
        ModelPrediction(
            model_name='lstm',
            prediction=0.7,
            confidence=0.90,
            features={'sequence_score': 0.7},
            metadata={'model_version': '1.0'}
        ),
        ModelPrediction(
            model_name='bert',
            prediction=0.75,
            confidence=0.88,
            features={'context_score': 0.75},
            metadata={'model_version': '1.0'}
        )
    ]
    
    # Test different fusion methods
    fusion_methods = [
        FusionMethod.WEIGHTED_AVERAGE,
        FusionMethod.DYNAMIC_WEIGHTING,
        FusionMethod.CONFIDENCE_WEIGHTED,
        FusionMethod.BAYESIAN_FUSION,
        FusionMethod.ADAPTIVE_FUSION
    ]
    
    text_features = {
        'char_count': 150,
        'word_count': 25,
        'sentiment_compound': 0.2,
        'flesch_reading_ease': 60.0
    }
    
    print("🧪 Testing different fusion methods...")
    
    for method in fusion_methods:
        fusion.fusion_method = method
        result = fusion.fuse_predictions(predictions, text_features)
        
        print(f"\n📊 {method.value.upper()}:")
        print(f"   Final Prediction: {result.final_prediction:.3f}")
        print(f"   Final Confidence: {result.final_confidence:.3f}")
        print(f"   Agreement Score: {result.agreement_score:.3f}")
        print(f"   Uncertainty: {result.uncertainty:.3f}")
        print(f"   Model Weights: {result.model_weights}")
    
    # Test with high disagreement
    print(f"\n🔀 Testing with high disagreement...")
    
    disagreeing_predictions = [
        ModelPrediction('svm', 0.9, 0.95, {}, {}),
        ModelPrediction('lstm', 0.3, 0.85, {}, {}),
        ModelPrediction('bert', 0.2, 0.88, {}, {})
    ]
    
    result = fusion.fuse_predictions(disagreeing_predictions, text_features)
    print(f"   Final Prediction: {result.final_prediction:.3f}")
    print(f"   Agreement Score: {result.agreement_score:.3f}")
    print(f"   Uncertainty: {result.uncertainty:.3f}")
    
    # Test performance update
    print(f"\n📈 Testing performance update...")
    fusion.update_model_performance('svm', 0.98, 0.96)
    print(f"   Updated SVM performance: {fusion.performance_tracker['svm']}")

if __name__ == "__main__":
    main()
