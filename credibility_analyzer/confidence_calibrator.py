"""
Confidence Calibrator for Credibility Analyzer
"""

import numpy as np
from typing import Dict, List

class ConfidenceCalibrator:
    """Calibrate confidence scores for better reliability"""
    
    def __init__(self):
        self.calibration_data = {
            'svm': {'bias': 0.05, 'scale': 0.95},
            'lstm': {'bias': -0.1, 'scale': 1.1},
            'bert': {'bias': 0.02, 'scale': 0.98}
        }
    
    def calibrate_confidence(self, raw_confidence: float, model_name: str) -> float:
        """Apply calibration to raw confidence score"""
        if model_name not in self.calibration_data:
            return raw_confidence
        
        calibration = self.calibration_data[model_name]
        calibrated = (raw_confidence + calibration['bias']) * calibration['scale']
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, calibrated))
    
    def ensemble_confidence(self, confidences: List[float], weights: List[float] = None) -> float:
        """Calculate ensemble confidence from multiple models"""
        if not confidences:
            return 0.5
        
        if weights is None:
            weights = [1.0 / len(confidences)] * len(confidences)
        
        # Weighted average
        weighted_confidence = sum(c * w for c, w in zip(confidences, weights))
        total_weight = sum(weights)
        
        return weighted_confidence / total_weight if total_weight > 0 else 0.5
