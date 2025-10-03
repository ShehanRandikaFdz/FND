"""
Verdict Agent - Multi-Agent AI System for Final Decision Making
Integrates with existing SVM, LSTM, and BERT models to provide final verdicts
with explainability, security, and responsible AI practices.
"""

import json
import logging
import hashlib
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import re
import requests
from flask import Flask, request, jsonify
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VerdictType(Enum):
    """Types of verdicts the agent can make"""
    REAL = "real"
    FAKE = "fake"
    UNCERTAIN = "uncertain"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"

class ConfidenceLevel(Enum):
    """Confidence levels for verdicts"""
    HIGH = "high"      # > 0.8
    MEDIUM = "medium"  # 0.5 - 0.8
    LOW = "low"       # < 0.5

@dataclass
class ModelResult:
    """Result from individual model"""
    model_name: str
    label: str
    confidence: float
    model_type: str
    accuracy: float

@dataclass
class Evidence:
    """Evidence supporting the verdict"""
    source: str
    content: str
    relevance_score: float
    citation: str

@dataclass
class VerdictResponse:
    """Final verdict response"""
    verdict: VerdictType
    confidence: float
    confidence_level: ConfidenceLevel
    reasoning: str
    evidence: List[Evidence]
    model_agreement: Dict[str, Any]
    audit_id: str
    timestamp: str
    processing_time_ms: int
    explainability: Dict[str, Any]

class VerdictAgent:
    """
    Verdict Agent - Final decision maker in the multi-agent system
    Integrates results from SVM, LSTM, BERT models with LLM reasoning
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the Verdict Agent"""
        self.config = config or self._default_config()
        self.llm_client = None
        self.ir_module = None
        self.nlp_pipeline = None
        self.audit_log = []
        
        # Initialize components
        self._setup_llm()
        self._setup_ir_module()
        self._setup_nlp_pipeline()
        
        logger.info("Verdict Agent initialized successfully")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for the agent"""
        return {
            "llm": {
                "provider": "openai",  # or "anthropic", "local"
                "model": "gpt-3.5-turbo",
                "temperature": 0.3,
                "max_tokens": 500
            },
            "security": {
                "max_input_length": 10000,
                "rate_limit_per_minute": 60,
                "enable_audit_logging": True
            },
            "verdict": {
                "confidence_threshold": 0.7,
                "uncertainty_threshold": 0.3,
                "ensemble_weights": {
                    "svm": 0.4,
                    "lstm": 0.3,
                    "bert": 0.3
                }
            }
        }
    
    def _setup_llm(self):
        """Setup LLM client for reasoning"""
        try:
            # For now, we'll use a mock LLM client
            # In production, this would connect to OpenAI, Anthropic, or local LLM
            self.llm_client = MockLLMClient(self.config["llm"])
            logger.info("LLM client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            self.llm_client = None
    
    def _setup_ir_module(self):
        """Setup Information Retrieval module"""
        try:
            self.ir_module = MockIRModule()
            logger.info("IR module initialized")
        except Exception as e:
            logger.error(f"Failed to initialize IR module: {e}")
            self.ir_module = None
    
    def _setup_nlp_pipeline(self):
        """Setup NLP pipeline for text processing"""
        try:
            self.nlp_pipeline = MockNLPPipeline()
            logger.info("NLP pipeline initialized")
        except Exception as e:
            logger.error(f"Failed to initialize NLP pipeline: {e}")
            self.nlp_pipeline = None
    
    def process_verdict(self, 
                       text: str, 
                       model_results: Dict[str, ModelResult],
                       user_context: Dict[str, Any] = None) -> VerdictResponse:
        """
        Process final verdict based on model results and additional analysis
        
        Args:
            text: Original text to analyze
            model_results: Results from SVM, LSTM, BERT models
            user_context: Additional context from user
            
        Returns:
            VerdictResponse with final decision and reasoning
        """
        start_time = time.time()
        audit_id = self._generate_audit_id(text, model_results)
        
        try:
            # Input validation and security checks
            self._validate_input(text, model_results)
            
            # Extract additional information using NLP pipeline
            nlp_analysis = self._analyze_with_nlp(text)
            
            # Gather evidence using IR module
            evidence = self._gather_evidence(text, nlp_analysis)
            
            # Calculate ensemble verdict
            ensemble_verdict = self._calculate_ensemble_verdict(model_results)
            
            # Use LLM for final reasoning if available
            llm_reasoning = self._get_llm_reasoning(text, model_results, nlp_analysis, evidence)
            
            # Make final verdict decision
            final_verdict = self._make_final_decision(
                ensemble_verdict, llm_reasoning, evidence, nlp_analysis
            )
            
            # Generate explainability information
            explainability = self._generate_explainability(
                model_results, nlp_analysis, evidence, final_verdict
            )
            
            # Calculate processing time
            processing_time = int((time.time() - start_time) * 1000)
            
            # Create response
            response = VerdictResponse(
                verdict=final_verdict["verdict"],
                confidence=final_verdict["confidence"],
                confidence_level=self._get_confidence_level(final_verdict["confidence"]),
                reasoning=final_verdict["reasoning"],
                evidence=evidence,
                model_agreement=self._calculate_model_agreement(model_results),
                audit_id=audit_id,
                timestamp=datetime.now().isoformat(),
                processing_time_ms=processing_time,
                explainability=explainability
            )
            
            # Log for audit
            self._log_verdict_decision(response, text, model_results)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing verdict: {e}")
            return self._create_error_response(str(e), audit_id)
    
    def _validate_input(self, text: str, model_results: Dict[str, ModelResult]):
        """Validate input for security and quality"""
        if not text or len(text.strip()) == 0:
            raise ValueError("Empty text provided")
        
        if len(text) > self.config["security"]["max_input_length"]:
            raise ValueError(f"Text too long: {len(text)} characters")
        
        if not model_results:
            raise ValueError("No model results provided")
        
        # Check for potential injection attacks
        if self._detect_injection_attempts(text):
            raise ValueError("Potential injection attempt detected")
    
    def _detect_injection_attempts(self, text: str) -> bool:
        """Detect potential prompt injection or malicious input"""
        suspicious_patterns = [
            r'ignore\s+previous\s+instructions',
            r'forget\s+everything',
            r'system\s+prompt',
            r'<script.*?>',
            r'javascript:',
            r'eval\(',
            r'exec\('
        ]
        
        text_lower = text.lower()
        for pattern in suspicious_patterns:
            if re.search(pattern, text_lower):
                return True
        return False
    
    def _analyze_with_nlp(self, text: str) -> Dict[str, Any]:
        """Analyze text using NLP pipeline"""
        if not self.nlp_pipeline:
            return {"entities": [], "summary": text[:200], "sentiment": "neutral"}
        
        return self.nlp_pipeline.analyze(text)
    
    def _gather_evidence(self, text: str, nlp_analysis: Dict[str, Any]) -> List[Evidence]:
        """Gather evidence using IR module"""
        if not self.ir_module:
            return []
        
        return self.ir_module.retrieve_evidence(text, nlp_analysis)
    
    def _calculate_ensemble_verdict(self, model_results: Dict[str, ModelResult]) -> Dict[str, Any]:
        """Calculate weighted ensemble verdict from model results"""
        weights = self.config["verdict"]["ensemble_weights"]
        total_weight = 0
        weighted_confidence = 0
        real_votes = 0
        fake_votes = 0
        
        for model_name, result in model_results.items():
            if model_name in weights:
                weight = weights[model_name]
                total_weight += weight
                
                if result.label.lower() == "real":
                    real_votes += weight
                else:
                    fake_votes += weight
                
                weighted_confidence += result.confidence * weight
        
        if total_weight == 0:
            return {"verdict": VerdictType.UNCERTAIN, "confidence": 0.0}
        
        # Determine verdict
        if real_votes > fake_votes:
            verdict = VerdictType.REAL
        elif fake_votes > real_votes:
            verdict = VerdictType.FAKE
        else:
            verdict = VerdictType.UNCERTAIN
        
        confidence = weighted_confidence / total_weight if total_weight > 0 else 0.0
        
        return {
            "verdict": verdict,
            "confidence": confidence,
            "real_votes": real_votes,
            "fake_votes": fake_votes
        }
    
    def _get_llm_reasoning(self, text: str, model_results: Dict[str, ModelResult], 
                          nlp_analysis: Dict[str, Any], evidence: List[Evidence]) -> Dict[str, Any]:
        """Get LLM-based reasoning for the verdict"""
        if not self.llm_client:
            return {"reasoning": "LLM not available", "confidence": 0.0}
        
        return self.llm_client.reason_verdict(text, model_results, nlp_analysis, evidence)
    
    def _make_final_decision(self, ensemble_verdict: Dict[str, Any], 
                           llm_reasoning: Dict[str, Any], 
                           evidence: List[Evidence],
                           nlp_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Make final verdict decision combining all information"""
        
        # Base decision on ensemble
        final_verdict = ensemble_verdict["verdict"]
        final_confidence = ensemble_verdict["confidence"]
        reasoning_parts = []
        
        # Add LLM reasoning if available
        if llm_reasoning and "reasoning" in llm_reasoning:
            reasoning_parts.append(f"LLM Analysis: {llm_reasoning['reasoning']}")
            # Adjust confidence based on LLM reasoning
            if "confidence" in llm_reasoning:
                final_confidence = (final_confidence + llm_reasoning["confidence"]) / 2
        
        # Add evidence-based reasoning
        if evidence:
            evidence_summary = f"Found {len(evidence)} pieces of supporting evidence"
            reasoning_parts.append(evidence_summary)
        
        # Add NLP analysis
        if nlp_analysis.get("sentiment"):
            reasoning_parts.append(f"Sentiment analysis: {nlp_analysis['sentiment']}")
        
        # Handle uncertainty
        if final_confidence < self.config["verdict"]["uncertainty_threshold"]:
            final_verdict = VerdictType.UNCERTAIN
            reasoning_parts.append("Low confidence across all models")
        
        # Combine reasoning
        reasoning = ". ".join(reasoning_parts) if reasoning_parts else "Standard ensemble analysis"
        
        return {
            "verdict": final_verdict,
            "confidence": final_confidence,
            "reasoning": reasoning
        }
    
    def _generate_explainability(self, model_results: Dict[str, ModelResult], 
                               nlp_analysis: Dict[str, Any], 
                               evidence: List[Evidence],
                               final_verdict: Dict[str, Any]) -> Dict[str, Any]:
        """Generate explainability information"""
        return {
            "model_contributions": {
                name: {
                    "label": result.label,
                    "confidence": result.confidence,
                    "weight": self.config["verdict"]["ensemble_weights"].get(name, 0)
                }
                for name, result in model_results.items()
            },
            "nlp_features": nlp_analysis,
            "evidence_count": len(evidence),
            "decision_factors": [
                "Ensemble model agreement",
                "LLM reasoning" if self.llm_client else "Standard analysis",
                "Evidence quality" if evidence else "No external evidence"
            ]
        }
    
    def _calculate_model_agreement(self, model_results: Dict[str, ModelResult]) -> Dict[str, Any]:
        """Calculate agreement between models"""
        labels = [result.label.lower() for result in model_results.values()]
        real_count = labels.count("real")
        fake_count = labels.count("fake")
        total = len(labels)
        
        return {
            "agreement_percentage": max(real_count, fake_count) / total * 100 if total > 0 else 0,
            "real_votes": real_count,
            "fake_votes": fake_count,
            "total_models": total
        }
    
    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Get confidence level from numeric confidence"""
        if confidence > 0.8:
            return ConfidenceLevel.HIGH
        elif confidence > 0.5:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def _generate_audit_id(self, text: str, model_results: Dict[str, ModelResult]) -> str:
        """Generate unique audit ID for tracking"""
        # Convert ModelResult objects to dict for JSON serialization
        model_results_dict = {
            name: {
                "label": result.label,
                "confidence": result.confidence,
                "model_type": result.model_type,
                "accuracy": result.accuracy
            }
            for name, result in model_results.items()
        }
        content = f"{text}{json.dumps(model_results_dict, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _log_verdict_decision(self, response: VerdictResponse, text: str, model_results: Dict[str, ModelResult]):
        """Log verdict decision for audit trail"""
        if self.config["security"]["enable_audit_logging"]:
            log_entry = {
                "audit_id": response.audit_id,
                "timestamp": response.timestamp,
                "verdict": response.verdict.value,
                "confidence": response.confidence,
                "processing_time_ms": response.processing_time_ms,
                "text_hash": hashlib.sha256(text.encode()).hexdigest()[:16],
                "model_results": {name: {"label": result.label, "confidence": result.confidence} 
                               for name, result in model_results.items()}
            }
            self.audit_log.append(log_entry)
            logger.info(f"Verdict logged: {response.audit_id}")
    
    def _create_error_response(self, error_message: str, audit_id: str) -> VerdictResponse:
        """Create error response"""
        return VerdictResponse(
            verdict=VerdictType.UNCERTAIN,
            confidence=0.0,
            confidence_level=ConfidenceLevel.LOW,
            reasoning=f"Error: {error_message}",
            evidence=[],
            model_agreement={},
            audit_id=audit_id,
            timestamp=datetime.now().isoformat(),
            processing_time_ms=0,
            explainability={"error": error_message}
        )


# Mock classes for initial implementation
class MockLLMClient:
    """Mock LLM client for testing"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def reason_verdict(self, text: str, model_results: Dict[str, ModelResult], 
                      nlp_analysis: Dict[str, Any], evidence: List[Evidence]) -> Dict[str, Any]:
        """Mock LLM reasoning"""
        return {
            "reasoning": "Mock LLM analysis: Based on the ensemble results and available evidence, this appears to be a standard news article with typical characteristics.",
            "confidence": 0.75
        }

class MockIRModule:
    """Mock Information Retrieval module"""
    
    def retrieve_evidence(self, text: str, nlp_analysis: Dict[str, Any]) -> List[Evidence]:
        """Mock evidence retrieval"""
        return [
            Evidence(
                source="Mock Database",
                content="Sample evidence supporting the analysis",
                relevance_score=0.8,
                citation="Mock Citation 2024"
            )
        ]

class MockNLPPipeline:
    """Mock NLP pipeline"""
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """Mock NLP analysis"""
        return {
            "entities": ["Mock Entity 1", "Mock Entity 2"],
            "summary": text[:100] + "...",
            "sentiment": "neutral",
            "key_phrases": ["mock phrase 1", "mock phrase 2"]
        }


if __name__ == "__main__":
    # Test the Verdict Agent
    agent = VerdictAgent()
    
    # Mock model results
    mock_results = {
        "svm": ModelResult("SVM", "Real", 0.95, "Traditional ML", 0.9959),
        "lstm": ModelResult("LSTM", "Real", 0.88, "Deep Learning", 0.9890),
        "bert": ModelResult("BERT", "Fake", 0.75, "Transformer", 0.9750)
    }
    
    # Test verdict processing
    test_text = "This is a test news article about technology developments."
    response = agent.process_verdict(test_text, mock_results)
    
    print("Verdict Agent Test Results:")
    print(f"Verdict: {response.verdict.value}")
    print(f"Confidence: {response.confidence:.3f}")
    print(f"Reasoning: {response.reasoning}")
    print(f"Audit ID: {response.audit_id}")
