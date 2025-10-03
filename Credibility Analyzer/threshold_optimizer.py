"""
Threshold Optimization for Credibility Analysis
Optimizes decision thresholds for fake/real/uncertain classifications.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import precision_recall_curve, roc_curve, f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from credibility_analyzer import CredibilityAnalyzer
import joblib
import warnings
warnings.filterwarnings('ignore')

class ThresholdOptimizer:
    """
    Optimizes decision thresholds for credibility analysis.
    """
    
    def __init__(self, analyzer: CredibilityAnalyzer):
        self.analyzer = analyzer
        self.optimal_thresholds = None
        self.threshold_analysis = None
        
    def load_test_data(self, fake_csv_path: str = 'Fake.csv', true_csv_path: str = 'True.csv') -> Tuple[pd.DataFrame, np.ndarray]:
        """Load test data for threshold optimization."""
        print("📊 Loading test data for threshold optimization...")
        
        # Load fake news data
        fake_df = pd.read_csv(fake_csv_path)
        fake_df['label'] = 0  # Fake = 0
        fake_df['text'] = fake_df['title'] + ' ' + fake_df['text']
        
        # Load true news data  
        true_df = pd.read_csv(true_csv_path)
        true_df['label'] = 1  # Real = 1
        true_df['text'] = true_df['title'] + ' ' + true_df['text']
        
        # Combine datasets
        combined_df = pd.concat([fake_df, true_df], ignore_index=True)
        
        # Shuffle data
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Use a subset for optimization (to avoid long processing time)
        sample_size = min(1000, len(combined_df))
        combined_df = combined_df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        
        print(f"✅ Loaded {len(combined_df)} samples ({len(fake_df)} fake, {len(true_df)} real)")
        
        return combined_df, combined_df['label'].values
    
    def get_predictions_with_confidence(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions and confidence scores for threshold optimization."""
        print("🔍 Getting predictions and confidence scores...")
        
        predictions = []
        confidence_scores = []
        
        for i, text in enumerate(texts):
            if i % 100 == 0:
                print(f"   Processed {i}/{len(texts)} samples...")
            
            try:
                result = self.analyzer.analyze_credibility(text)
                
                # Extract credibility score (0-1, where 1 = real)
                credibility_score = result.get('credibility_score', 0.5)
                predictions.append(credibility_score)
                
                # Extract confidence score
                confidence = result.get('confidence', 0.5)
                confidence_scores.append(confidence)
                
            except Exception as e:
                print(f"   Warning: Error processing sample {i}: {e}")
                predictions.append(0.5)  # Neutral prediction
                confidence_scores.append(0.5)  # Low confidence
        
        return np.array(predictions), np.array(confidence_scores)
    
    def optimize_binary_threshold(self, predictions: np.ndarray, true_labels: np.ndarray) -> Dict[str, float]:
        """Optimize threshold for binary fake/real classification."""
        print("🎯 Optimizing binary classification threshold...")
        
        # Generate threshold candidates
        thresholds = np.linspace(0.1, 0.9, 81)  # 0.1 to 0.9 in steps of 0.01
        
        best_threshold = 0.5
        best_f1 = 0.0
        best_accuracy = 0.0
        
        threshold_results = []
        
        for threshold in thresholds:
            # Convert probabilities to binary predictions
            binary_preds = (predictions >= threshold).astype(int)
            
            # Calculate metrics
            accuracy = accuracy_score(true_labels, binary_preds)
            f1 = f1_score(true_labels, binary_preds, average='weighted')
            
            # Calculate precision and recall for each class
            tn, fp, fn, tp = confusion_matrix(true_labels, binary_preds).ravel()
            
            precision_real = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall_real = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision_fake = tn / (tn + fn) if (tn + fn) > 0 else 0
            recall_fake = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            threshold_results.append({
                'threshold': threshold,
                'accuracy': accuracy,
                'f1_score': f1,
                'precision_real': precision_real,
                'recall_real': recall_real,
                'precision_fake': precision_fake,
                'recall_fake': recall_fake,
                'true_positives': tp,
                'true_negatives': tn,
                'false_positives': fp,
                'false_negatives': fn
            })
            
            # Update best threshold based on F1 score
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_accuracy = accuracy
        
        return {
            'optimal_threshold': best_threshold,
            'best_f1_score': best_f1,
            'best_accuracy': best_accuracy,
            'all_results': threshold_results
        }
    
    def optimize_three_way_thresholds(self, predictions: np.ndarray, confidence_scores: np.ndarray, 
                                    true_labels: np.ndarray) -> Dict[str, any]:
        """Optimize thresholds for three-way classification (fake/uncertain/real)."""
        print("🎯 Optimizing three-way classification thresholds...")
        
        # Generate threshold candidates
        credibility_thresholds = np.linspace(0.2, 0.8, 31)  # 0.2 to 0.8 in steps of 0.02
        confidence_thresholds = np.linspace(0.5, 0.9, 21)   # 0.5 to 0.9 in steps of 0.02
        
        best_credibility_threshold = 0.5
        best_confidence_threshold = 0.7
        best_score = 0.0
        
        results = []
        
        for cred_thresh in credibility_thresholds:
            for conf_thresh in confidence_thresholds:
                # Apply three-way classification
                three_way_preds = self._apply_three_way_classification(
                    predictions, confidence_scores, cred_thresh, conf_thresh
                )
                
                # Calculate metrics for three-way classification
                score = self._evaluate_three_way_classification(true_labels, three_way_preds)
                
                results.append({
                    'credibility_threshold': cred_thresh,
                    'confidence_threshold': conf_thresh,
                    'score': score
                })
                
                if score > best_score:
                    best_score = score
                    best_credibility_threshold = cred_thresh
                    best_confidence_threshold = conf_thresh
        
        return {
            'optimal_credibility_threshold': best_credibility_threshold,
            'optimal_confidence_threshold': best_confidence_threshold,
            'best_score': best_score,
            'all_results': results
        }
    
    def _apply_three_way_classification(self, predictions: np.ndarray, confidence_scores: np.ndarray,
                                      credibility_threshold: float, confidence_threshold: float) -> np.ndarray:
        """Apply three-way classification based on thresholds."""
        three_way_preds = np.zeros(len(predictions))
        
        for i, (pred, conf) in enumerate(zip(predictions, confidence_scores)):
            if conf < confidence_threshold:
                three_way_preds[i] = 0.5  # Uncertain
            elif pred >= credibility_threshold:
                three_way_preds[i] = 1.0  # Real
            else:
                three_way_preds[i] = 0.0  # Fake
        
        return three_way_preds
    
    def _evaluate_three_way_classification(self, true_labels: np.ndarray, three_way_preds: np.ndarray) -> float:
        """Evaluate three-way classification performance."""
        # Convert true labels to three-way (assume uncertain = 0.5)
        # For evaluation, we'll focus on the clear cases (fake=0, real=1)
        
        # Find clear predictions (not uncertain)
        clear_mask = three_way_preds != 0.5
        clear_true = true_labels[clear_mask]
        clear_preds = three_way_preds[clear_mask]
        
        if len(clear_true) == 0:
            return 0.0
        
        # Calculate accuracy on clear predictions
        accuracy = accuracy_score(clear_true, clear_preds)
        
        # Penalize for too many uncertain predictions
        uncertainty_rate = np.mean(three_way_preds == 0.5)
        uncertainty_penalty = uncertainty_rate * 0.2  # Penalty for excessive uncertainty
        
        return accuracy - uncertainty_penalty
    
    def analyze_threshold_sensitivity(self, predictions: np.ndarray, true_labels: np.ndarray,
                                    confidence_scores: np.ndarray) -> Dict[str, any]:
        """Analyze sensitivity of predictions to threshold changes."""
        print("📈 Analyzing threshold sensitivity...")
        
        # Test different threshold ranges
        thresholds = np.linspace(0.3, 0.7, 21)
        
        sensitivity_results = []
        
        for threshold in thresholds:
            binary_preds = (predictions >= threshold).astype(int)
            
            # Calculate metrics
            accuracy = accuracy_score(true_labels, binary_preds)
            
            # Calculate class-specific metrics
            tn, fp, fn, tp = confusion_matrix(true_labels, binary_preds).ravel()
            
            precision_real = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall_real = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision_fake = tn / (tn + fn) if (tn + fn) > 0 else 0
            recall_fake = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # Calculate confidence-based metrics
            high_conf_mask = confidence_scores >= 0.8
            if np.sum(high_conf_mask) > 0:
                high_conf_accuracy = accuracy_score(
                    true_labels[high_conf_mask], 
                    binary_preds[high_conf_mask]
                )
            else:
                high_conf_accuracy = 0.0
            
            sensitivity_results.append({
                'threshold': threshold,
                'accuracy': accuracy,
                'precision_real': precision_real,
                'recall_real': recall_real,
                'precision_fake': precision_fake,
                'recall_fake': recall_fake,
                'high_confidence_accuracy': high_conf_accuracy,
                'high_confidence_count': np.sum(high_conf_mask)
            })
        
        return sensitivity_results
    
    def optimize_all_thresholds(self, test_data_path: str = None) -> Dict[str, any]:
        """Run complete threshold optimization."""
        print("🎯 Starting Complete Threshold Optimization")
        print("=" * 60)
        
        # Load test data
        if test_data_path:
            df, true_labels = self.load_test_data(test_data_path)
        else:
            df, true_labels = self.load_test_data()
        
        texts = df['text'].tolist()
        
        # Get predictions and confidence scores
        predictions, confidence_scores = self.get_predictions_with_confidence(texts)
        
        # Optimize binary classification threshold
        binary_results = self.optimize_binary_threshold(predictions, true_labels)
        
        # Optimize three-way classification thresholds
        three_way_results = self.optimize_three_way_thresholds(predictions, confidence_scores, true_labels)
        
        # Analyze threshold sensitivity
        sensitivity_results = self.analyze_threshold_sensitivity(predictions, true_labels, confidence_scores)
        
        # Compile results
        optimization_results = {
            'binary_classification': binary_results,
            'three_way_classification': three_way_results,
            'threshold_sensitivity': sensitivity_results,
            'data_summary': {
                'total_samples': len(texts),
                'fake_samples': np.sum(true_labels == 0),
                'real_samples': np.sum(true_labels == 1),
                'avg_prediction': np.mean(predictions),
                'avg_confidence': np.mean(confidence_scores)
            }
        }
        
        # Save optimal thresholds
        self.optimal_thresholds = {
            'binary_threshold': binary_results['optimal_threshold'],
            'credibility_threshold': three_way_results['optimal_credibility_threshold'],
            'confidence_threshold': three_way_results['optimal_confidence_threshold']
        }
        
        # Save results
        joblib.dump(optimization_results, 'models/threshold_optimization_results.pkl')
        joblib.dump(self.optimal_thresholds, 'models/optimal_thresholds.pkl')
        
        print(f"\n✅ Threshold optimization completed!")
        print(f"📊 Optimal binary threshold: {self.optimal_thresholds['binary_threshold']:.3f}")
        print(f"📊 Optimal credibility threshold: {self.optimal_thresholds['credibility_threshold']:.3f}")
        print(f"📊 Optimal confidence threshold: {self.optimal_thresholds['confidence_threshold']:.3f}")
        
        return optimization_results
    
    def visualize_threshold_analysis(self, results: Dict[str, any]) -> None:
        """Create visualizations for threshold analysis."""
        print("📊 Creating threshold analysis visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Credibility Analysis Threshold Optimization', fontsize=16)
        
        # Binary classification threshold analysis
        binary_results = results['binary_classification']['all_results']
        thresholds = [r['threshold'] for r in binary_results]
        accuracies = [r['accuracy'] for r in binary_results]
        f1_scores = [r['f1_score'] for r in binary_results]
        
        axes[0, 0].plot(thresholds, accuracies, 'b-', label='Accuracy', linewidth=2)
        axes[0, 0].plot(thresholds, f1_scores, 'r-', label='F1 Score', linewidth=2)
        axes[0, 0].axvline(results['binary_classification']['optimal_threshold'], 
                          color='green', linestyle='--', label='Optimal Threshold')
        axes[0, 0].set_xlabel('Threshold')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Binary Classification Threshold Optimization')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Three-way classification heatmap
        three_way_results = results['three_way_classification']['all_results']
        cred_thresholds = sorted(set(r['credibility_threshold'] for r in three_way_results))
        conf_thresholds = sorted(set(r['confidence_threshold'] for r in three_way_results))
        
        score_matrix = np.zeros((len(conf_thresholds), len(cred_thresholds)))
        for r in three_way_results:
            i = conf_thresholds.index(r['confidence_threshold'])
            j = cred_thresholds.index(r['credibility_threshold'])
            score_matrix[i, j] = r['score']
        
        im = axes[0, 1].imshow(score_matrix, cmap='viridis', aspect='auto')
        axes[0, 1].set_xticks(range(len(cred_thresholds)))
        axes[0, 1].set_xticklabels([f'{t:.2f}' for t in cred_thresholds], rotation=45)
        axes[0, 1].set_yticks(range(len(conf_thresholds)))
        axes[0, 1].set_yticklabels([f'{t:.2f}' for t in conf_thresholds])
        axes[0, 1].set_xlabel('Credibility Threshold')
        axes[0, 1].set_ylabel('Confidence Threshold')
        axes[0, 1].set_title('Three-Way Classification Optimization')
        plt.colorbar(im, ax=axes[0, 1])
        
        # Threshold sensitivity analysis
        sensitivity_results = results['threshold_sensitivity']
        sens_thresholds = [r['threshold'] for r in sensitivity_results]
        sens_accuracies = [r['accuracy'] for r in sensitivity_results]
        high_conf_accuracies = [r['high_confidence_accuracy'] for r in sensitivity_results]
        
        axes[1, 0].plot(sens_thresholds, sens_accuracies, 'b-', label='Overall Accuracy', linewidth=2)
        axes[1, 0].plot(sens_thresholds, high_conf_accuracies, 'r-', label='High Confidence Accuracy', linewidth=2)
        axes[1, 0].set_xlabel('Threshold')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Threshold Sensitivity Analysis')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Precision-Recall curves for real and fake
        precision_real = [r['precision_real'] for r in binary_results]
        recall_real = [r['recall_real'] for r in binary_results]
        precision_fake = [r['precision_fake'] for r in binary_results]
        recall_fake = [r['recall_fake'] for r in binary_results]
        
        axes[1, 1].plot(recall_real, precision_real, 'b-', label='Real News', linewidth=2)
        axes[1, 1].plot(recall_fake, precision_fake, 'r-', label='Fake News', linewidth=2)
        axes[1, 1].set_xlabel('Recall')
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].set_title('Precision-Recall Curves')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('models/threshold_optimization_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ Visualization saved to 'models/threshold_optimization_analysis.png'")

def main():
    """Test the threshold optimizer."""
    print("🎯 Testing Threshold Optimizer")
    print("=" * 50)
    
    # Initialize components
    analyzer = CredibilityAnalyzer()
    optimizer = ThresholdOptimizer(analyzer)
    
    # Run threshold optimization
    results = optimizer.optimize_all_thresholds()
    
    # Create visualizations
    try:
        optimizer.visualize_threshold_analysis(results)
    except Exception as e:
        print(f"Warning: Could not create visualizations: {e}")
    
    # Print summary
    print(f"\n📊 Threshold Optimization Summary")
    print("=" * 60)
    
    binary = results['binary_classification']
    three_way = results['three_way_classification']
    
    print(f"🎯 Binary Classification:")
    print(f"   Optimal Threshold: {binary['optimal_threshold']:.3f}")
    print(f"   Best F1 Score: {binary['best_f1_score']:.3f}")
    print(f"   Best Accuracy: {binary['best_accuracy']:.3f}")
    
    print(f"\n🎯 Three-Way Classification:")
    print(f"   Credibility Threshold: {three_way['optimal_credibility_threshold']:.3f}")
    print(f"   Confidence Threshold: {three_way['optimal_confidence_threshold']:.3f}")
    print(f"   Best Score: {three_way['best_score']:.3f}")
    
    print(f"\n📈 Data Summary:")
    summary = results['data_summary']
    print(f"   Total Samples: {summary['total_samples']}")
    print(f"   Fake/Real Ratio: {summary['fake_samples']}/{summary['real_samples']}")
    print(f"   Avg Prediction: {summary['avg_prediction']:.3f}")
    print(f"   Avg Confidence: {summary['avg_confidence']:.3f}")

if __name__ == "__main__":
    main()
