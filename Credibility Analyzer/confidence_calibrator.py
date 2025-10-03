"""
Confidence Calibration and Uncertainty Quantification
Enhances the credibility analyzer with better confidence scoring.
"""

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import brier_score_loss, log_loss
import matplotlib.pyplot as plt
import pickle
import os
from typing import Dict, List, Tuple, Optional
from credibility_analyzer import CredibilityAnalyzer

class ConfidenceCalibrator:
    """
    Calibrates confidence scores and provides uncertainty quantification.
    """
    
    def __init__(self, analyzer: CredibilityAnalyzer):
        self.analyzer = analyzer
        self.calibrators = {}
        self.uncertainty_bands = {
            'very_certain': (0.9, 1.0),
            'certain': (0.8, 0.9),
            'moderate': (0.6, 0.8),
            'uncertain': (0.4, 0.6),
            'very_uncertain': (0.0, 0.4)
        }
        
    def load_calibration_data(self, data_file: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """Load or generate calibration data."""
        if data_file and os.path.exists(data_file):
            data = pd.read_csv(data_file)
            return data['text'].values, data['label'].values
        
        # Use existing dataset for calibration
        fake_df = pd.read_csv('Fake.csv')
        true_df = pd.read_csv('True.csv')
        
        fake_df['class'] = 0
        true_df['class'] = 1
        
        combined_df = pd.concat([fake_df, true_df], ignore_index=True)
        combined_df = combined_df.drop(['title', 'subject', 'date'], axis=1)
        
        # Sample for calibration (use smaller subset for speed)
        sample_size = min(2000, len(combined_df))
        sample_df = combined_df.sample(n=sample_size, random_state=42)
        
        return sample_df['text'].values, sample_df['class'].values
    
    def generate_predictions_for_calibration(self, texts: np.ndarray, labels: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate predictions from all models for calibration."""
        print("🔧 Generating predictions for calibration...")
        
        predictions = {
            'svm': [],
            'lstm': [],
            'bert': [],
            'ensemble': []
        }
        
        confidences = {
            'svm': [],
            'lstm': [],
            'bert': [],
            'ensemble': []
        }
        
        for i, text in enumerate(texts):
            if i % 100 == 0:
                print(f"   Processing {i}/{len(texts)} texts...")
            
            try:
                # Get individual model predictions
                if 'svm' in self.analyzer.models:
                    svm_pred = self.analyzer.predict_svm(text)
                    predictions['svm'].append(svm_pred['prediction'])
                    confidences['svm'].append(svm_pred['confidence'])
                
                if 'lstm' in self.analyzer.models:
                    lstm_pred = self.analyzer.predict_lstm(text)
                    predictions['lstm'].append(lstm_pred['prediction'])
                    confidences['lstm'].append(lstm_pred['confidence'])
                
                if 'bert' in self.analyzer.models:
                    bert_pred = self.analyzer.predict_bert(text)
                    predictions['bert'].append(bert_pred['prediction'])
                    confidences['bert'].append(bert_pred['confidence'])
                
                # Get ensemble prediction
                ensemble_result = self.analyzer.ensemble_predict(text)
                predictions['ensemble'].append(ensemble_result['credibility_score'])
                confidences['ensemble'].append(ensemble_result['confidence'])
                
            except Exception as e:
                print(f"   Error processing text {i}: {e}")
                # Add default values for failed predictions
                for model in predictions.keys():
                    predictions[model].append(0.5)
                    confidences[model].append(0.5)
        
        # Convert to numpy arrays
        for model in predictions.keys():
            predictions[model] = np.array(predictions[model])
            confidences[model] = np.array(confidences[model])
        
        return predictions, confidences, labels
    
    def calibrate_confidence(self, predictions: Dict[str, np.ndarray], 
                           confidences: Dict[str, np.ndarray], 
                           labels: np.ndarray):
        """Calibrate confidence scores using isotonic regression."""
        print("📊 Calibrating confidence scores...")
        
        for model_name in predictions.keys():
            if len(predictions[model_name]) == 0:
                continue
                
            print(f"   Calibrating {model_name}...")
            
            # Use isotonic regression for calibration
            calibrator = IsotonicRegression(out_of_bounds='clip')
            
            # Fit calibrator on confidence scores vs actual accuracy
            # Create binary accuracy: 1 if prediction matches label, 0 otherwise
            binary_preds = (predictions[model_name] > 0.5).astype(int)
            accuracy_per_prediction = (binary_preds == labels).astype(float)
            
            # Fit calibrator
            calibrator.fit(confidences[model_name], accuracy_per_prediction)
            self.calibrators[model_name] = calibrator
            
            # Evaluate calibration
            calibrated_conf = calibrator.predict(confidences[model_name])
            
            print(f"      Original confidence range: [{confidences[model_name].min():.3f}, {confidences[model_name].max():.3f}]")
            print(f"      Calibrated confidence range: [{calibrated_conf.min():.3f}, {calibrated_conf.max():.3f}]")
            print(f"      Mean accuracy: {accuracy_per_prediction.mean():.3f}")
            print(f"      Mean calibrated confidence: {calibrated_conf.mean():.3f}")
    
    def get_calibrated_confidence(self, model_name: str, raw_confidence: float) -> float:
        """Get calibrated confidence score."""
        if model_name not in self.calibrators:
            return raw_confidence
        
        calibrated = self.calibrators[model_name].predict([raw_confidence])[0]
        return float(np.clip(calibrated, 0.0, 1.0))
    
    def get_uncertainty_band(self, confidence: float) -> str:
        """Classify confidence into uncertainty bands."""
        for band_name, (low, high) in self.uncertainty_bands.items():
            if low <= confidence < high:
                return band_name
        return 'very_uncertain'
    
    def compute_prediction_interval(self, predictions: List[float], confidence_level: float = 0.95) -> Tuple[float, float]:
        """Compute prediction interval for ensemble predictions."""
        if len(predictions) < 2:
            return (0.0, 1.0)
        
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(predictions, lower_percentile)
        upper_bound = np.percentile(predictions, upper_percentile)
        
        return (float(lower_bound), float(upper_bound))
    
    def enhanced_credibility_analysis(self, text: str) -> Dict[str, any]:
        """Enhanced analysis with calibrated confidence and uncertainty quantification."""
        # Get base analysis
        base_result = self.analyzer.analyze_credibility(text)
        
        if 'error' in base_result:
            return base_result
        
        # Calibrate confidences
        calibrated_confidences = {}
        individual_preds = base_result.get('individual_predictions', {})
        
        for model_name, pred_data in individual_preds.items():
            raw_conf = pred_data.get('confidence', 0.5)
            calibrated_conf = self.get_calibrated_confidence(model_name, raw_conf)
            calibrated_confidences[model_name] = calibrated_conf
        
        # Calibrate ensemble confidence
        ensemble_raw_conf = base_result.get('confidence', 0.5)
        ensemble_calibrated_conf = self.get_calibrated_confidence('ensemble', ensemble_raw_conf)
        
        # Compute prediction interval
        individual_scores = [pred['prediction'] for pred in individual_preds.values()]
        prediction_interval = self.compute_prediction_interval(individual_scores)
        
        # Determine uncertainty band
        uncertainty_band = self.get_uncertainty_band(ensemble_calibrated_conf)
        
        # Enhanced result
        enhanced_result = base_result.copy()
        enhanced_result.update({
            'calibrated_confidence': ensemble_calibrated_conf,
            'calibrated_confidences_by_model': calibrated_confidences,
            'uncertainty_band': uncertainty_band,
            'prediction_interval': prediction_interval,
            'confidence_calibration_applied': True
        })
        
        return enhanced_result
    
    def save_calibrators(self, filepath: str = 'models/confidence_calibrators.pkl'):
        """Save trained calibrators."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.calibrators, f)
        print(f"💾 Calibrators saved to {filepath}")
    
    def load_calibrators(self, filepath: str = 'models/confidence_calibrators.pkl'):
        """Load trained calibrators."""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.calibrators = pickle.load(f)
            print(f"📁 Calibrators loaded from {filepath}")
            return True
        return False
    
    def plot_calibration_curve(self, predictions: Dict[str, np.ndarray], 
                              confidences: Dict[str, np.ndarray], 
                              labels: np.ndarray, 
                              save_path: str = 'calibration_curves.png'):
        """Plot calibration curves for all models."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for i, (model_name, conf_scores) in enumerate(confidences.items()):
            if i >= 4:  # Only plot first 4 models
                break
                
            if len(conf_scores) == 0:
                continue
            
            # Compute calibration curve
            binary_preds = (predictions[model_name] > 0.5).astype(int)
            fraction_of_positives, mean_predicted_value = calibration_curve(
                labels, conf_scores, n_bins=10
            )
            
            # Plot
            axes[i].plot(mean_predicted_value, fraction_of_positives, "s-", label=f"{model_name}")
            axes[i].plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
            axes[i].set_xlabel("Mean Predicted Probability")
            axes[i].set_ylabel("Fraction of Positives")
            axes[i].set_title(f"Calibration Curve - {model_name.upper()}")
            axes[i].legend()
            axes[i].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Calibration curves saved to {save_path}")

def main():
    """Test confidence calibration."""
    print("📊 Testing Confidence Calibration")
    print("=" * 50)
    
    # Initialize analyzer and calibrator
    analyzer = CredibilityAnalyzer()
    calibrator = ConfidenceCalibrator(analyzer)
    
    # Try to load existing calibrators
    if not calibrator.load_calibrators():
        print("🔧 Training new calibrators...")
        
        # Load calibration data
        texts, labels = calibrator.load_calibration_data()
        print(f"📚 Loaded {len(texts)} texts for calibration")
        
        # Generate predictions
        predictions, confidences, labels = calibrator.generate_predictions_for_calibration(texts, labels)
        
        # Calibrate confidence scores
        calibrator.calibrate_confidence(predictions, confidences, labels)
        
        # Save calibrators
        calibrator.save_calibrators()
        
        # Plot calibration curves
        calibrator.plot_calibration_curve(predictions, confidences, labels)
    
    # Test enhanced analysis
    print("\n🧪 Testing Enhanced Analysis")
    print("-" * 30)
    
    test_texts = [
        "Reuters reports that the Federal Reserve announced new interest rate policies today.",
        "BREAKING: Scientists discover aliens living in your backyard! Click here for shocking photos!",
        "NASA announces new evidence of water on Mars."
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n📰 Test {i}: {text[:50]}...")
        result = calibrator.enhanced_credibility_analysis(text)
        
        if 'error' not in result:
            print(f"   🎯 Credibility Score: {result['credibility_score']:.4f}")
            print(f"   📊 Raw Confidence: {result['confidence']:.4f}")
            print(f"   🔧 Calibrated Confidence: {result['calibrated_confidence']:.4f}")
            print(f"   🎭 Uncertainty Band: {result['uncertainty_band']}")
            print(f"   📏 Prediction Interval: [{result['prediction_interval'][0]:.3f}, {result['prediction_interval'][1]:.3f}]")
        else:
            print(f"   ❌ Error: {result['error']}")

if __name__ == "__main__":
    main()
