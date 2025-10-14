"""
Verdict Agent for Flask Application
Multi-agent AI system for final decision-making
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VerdictType(Enum):
    """Types of verdicts"""
    TRUE = "true"
    FALSE = "false"
    MISLEADING = "misleading"
    UNCERTAIN = "uncertain"

@dataclass
class ModelResult:
    """Result from a single model"""
    model_name: str
    label: str  # 'true', 'false', 'misleading'
    confidence: float
    model_type: str
    accuracy: float

class VerdictAgent:
    """Multi-agent system for generating final verdicts"""
    
    def __init__(self):
        self.llm_client = self._initialize_llm_client()
        self.ir_module = self._initialize_ir_module()
        self.nlp_pipeline = self._initialize_nlp_pipeline()
        logger.info("Verdict Agent initialized successfully")
    
    def _initialize_llm_client(self):
        """Initialize LLM client (placeholder)"""
        logger.info("LLM client initialized")
        return None
    
    def _initialize_ir_module(self):
        """Initialize information retrieval module"""
        logger.info("IR module initialized")
        return None
    
    def _initialize_nlp_pipeline(self):
        """Initialize NLP pipeline"""
        logger.info("NLP pipeline initialized")
        return None
    
    def generate_verdict(self, text: str, model_results: List[ModelResult]) -> Dict:
        """Generate final verdict from model results"""
        try:
            # Analyze model consensus
            consensus = self._analyze_consensus(model_results)
            
            # Generate explanation
            explanation = self._generate_explanation(text, model_results, consensus)
            
            # Calculate final confidence
            confidence = self._calculate_final_confidence(model_results, consensus)
            
            return {
                'verdict': consensus['verdict'],
                'confidence': confidence,
                'explanation': explanation,
                'consensus_analysis': consensus,
                'model_results': [self._serialize_model_result(mr) for mr in model_results]
            }
            
        except Exception as e:
            logger.error(f"Error generating verdict: {e}")
            return {
                'verdict': VerdictType.UNCERTAIN.value,
                'confidence': 0.0,
                'explanation': f'Error in verdict generation: {str(e)}',
                'consensus_analysis': {},
                'error': str(e)
            }
    
    def _analyze_consensus(self, model_results: List[ModelResult]) -> Dict:
        """Analyze consensus among model results"""
        if not model_results:
            return {
                'verdict': VerdictType.UNCERTAIN.value,
                'agreement_level': 0.0,
                'majority_label': None,
                'confidence_variance': 1.0
            }
        
        # Count votes
        vote_counts = {}
        total_confidence = 0.0
        
        for result in model_results:
            label = result.label
            confidence = result.confidence
            
            if label not in vote_counts:
                vote_counts[label] = {'count': 0, 'confidence_sum': 0.0}
            
            vote_counts[label]['count'] += 1
            vote_counts[label]['confidence_sum'] += confidence
            total_confidence += confidence
        
        # Find majority
        majority_label = max(vote_counts.keys(), key=lambda k: vote_counts[k]['count'])
        majority_count = vote_counts[majority_label]['count']
        total_models = len(model_results)
        
        # Calculate agreement level
        agreement_level = majority_count / total_models
        
        # Calculate confidence variance
        avg_confidence = total_confidence / total_models
        confidence_variance = sum((r.confidence - avg_confidence) ** 2 for r in model_results) / total_models
        
        # Map to verdict type
        verdict = self._map_to_verdict_type(majority_label)
        
        return {
            'verdict': verdict.value,
            'agreement_level': agreement_level,
            'majority_label': majority_label,
            'confidence_variance': confidence_variance,
            'vote_counts': vote_counts
        }
    
    def _map_to_verdict_type(self, label: str) -> VerdictType:
        """Map model label to verdict type"""
        label_lower = label.lower()
        
        if label_lower in ['true', 'real', 'authentic']:
            return VerdictType.TRUE
        elif label_lower in ['false', 'fake', 'untrue']:
            return VerdictType.FALSE
        elif label_lower in ['misleading', 'partially_false', 'deceptive']:
            return VerdictType.MISLEADING
        else:
            return VerdictType.UNCERTAIN
    
    def _generate_explanation(self, text: str, model_results: List[ModelResult], consensus: Dict) -> str:
        """Generate human-readable explanation"""
        verdict = consensus['verdict']
        agreement_level = consensus['agreement_level']
        
        if agreement_level >= 0.8:
            confidence_desc = "high"
        elif agreement_level >= 0.6:
            confidence_desc = "moderate"
        else:
            confidence_desc = "low"
        
        # Base explanation
        if verdict == VerdictType.TRUE.value:
            explanation = f"The content appears to be TRUE with {confidence_desc} confidence. "
        elif verdict == VerdictType.FALSE.value:
            explanation = f"The content appears to be FALSE with {confidence_desc} confidence. "
        elif verdict == VerdictType.MISLEADING.value:
            explanation = f"The content appears to be MISLEADING with {confidence_desc} confidence. "
        else:
            explanation = f"The content authenticity is UNCERTAIN with {confidence_desc} confidence. "
        
        # Add model details
        model_names = [mr.model_name for mr in model_results]
        explanation += f"Analysis based on {len(model_results)} models: {', '.join(model_names)}. "
        
        # Add agreement details
        if agreement_level >= 0.8:
            explanation += "Strong consensus among models."
        elif agreement_level >= 0.6:
            explanation += "Moderate consensus among models."
        else:
            explanation += "Limited consensus among models - additional verification recommended."
        
        return explanation
    
    def _calculate_final_confidence(self, model_results: List[ModelResult], consensus: Dict) -> float:
        """Calculate final confidence score"""
        if not model_results:
            return 0.0
        
        # Weight by agreement level
        agreement_level = consensus['agreement_level']
        
        # Average confidence of models
        avg_confidence = sum(mr.confidence for mr in model_results) / len(model_results)
        
        # Adjust by consensus
        final_confidence = avg_confidence * agreement_level
        
        return min(1.0, final_confidence)
    
    def _serialize_model_result(self, model_result: ModelResult) -> Dict:
        """Serialize model result for JSON response"""
        return {
            'model_name': model_result.model_name,
            'label': model_result.label,
            'confidence': model_result.confidence,
            'model_type': model_result.model_type,
            'accuracy': model_result.accuracy
        }
