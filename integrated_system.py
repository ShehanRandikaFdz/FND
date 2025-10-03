"""
Integrated Multi-Agent System
Combines existing SVM/LSTM/BERT models with Verdict Agent
Provides unified API for the complete system
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import time
import requests
from typing import Dict, Any, List
import json

# Import existing model predictor
from app_final import ModelPredictor

# Import verdict agent
from verdict_agent import VerdictAgent, ModelResult, VerdictType, ConfidenceLevel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize components
print("Initializing integrated multi-agent system...")

# Initialize existing model predictor
model_predictor = ModelPredictor()
print("✅ Model predictor initialized")

# Initialize verdict agent
verdict_agent = VerdictAgent()
print("✅ Verdict agent initialized")

# System configuration
SYSTEM_CONFIG = {
    "name": "Multi-Agent Fake News Detection System",
    "version": "2.0.0",
    "agents": ["SVM Agent", "LSTM Agent", "BERT Agent", "Verdict Agent"],
    "features": [
        "Multi-model ensemble",
        "LLM reasoning",
        "Evidence gathering",
        "Explainable AI",
        "Audit logging",
        "Security controls"
    ]
}

@app.route("/", methods=["GET"])
def home():
    """Homepage with system documentation"""
    return f"""
    <html>
    <head>
        <title>Multi-Agent Fake News Detection System</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #333; text-align: center; }}
            .agent-card {{ background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; border-left: 4px solid #007bff; }}
            .verdict-card {{ background: #e8f5e8; padding: 15px; border-radius: 5px; margin: 20px 0; border-left: 4px solid #28a745; }}
            .method {{ color: #28a745; font-weight: bold; }}
            .url {{ color: #007bff; font-family: monospace; }}
            .example {{ background: #e9ecef; padding: 10px; border-radius: 3px; margin: 10px 0; }}
            .status {{ color: #28a745; font-weight: bold; }}
            .accuracy {{ color: #dc3545; font-weight: bold; }}
            .neural {{ color: #6f42c1; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 Multi-Agent Fake News Detection System</h1>
            <p><span class="status">✅ System running with 4 intelligent agents!</span></p>
            
            <h2>System Architecture</h2>
            <div class="agent-card">
                <h3>🔍 Detection Agents (SVM, LSTM, BERT)</h3>
                <p>Traditional ML, Deep Learning, and Transformer models working in parallel</p>
                <ul>
                    <li><strong>SVM:</strong> 99.59% accuracy, fast inference</li>
                    <li><strong>LSTM:</strong> 98.90% accuracy, sequential patterns</li>
                    <li><strong>BERT:</strong> 97.50% accuracy, contextual understanding</li>
                </ul>
            </div>
            
            <div class="verdict-card">
                <h3>⚖️ Verdict Agent (NEW)</h3>
                <p>Final decision maker with LLM reasoning, evidence gathering, and explainability</p>
                <ul>
                    <li><strong>Ensemble Analysis:</strong> Weighted combination of all models</li>
                    <li><strong>LLM Reasoning:</strong> Advanced reasoning for complex cases</li>
                    <li><strong>Evidence Gathering:</strong> Information retrieval and fact-checking</li>
                    <li><strong>Explainable AI:</strong> Transparent decision-making process</li>
                    <li><strong>Security:</strong> Input validation, rate limiting, audit logging</li>
                </ul>
            </div>
            
            <h2>API Endpoints</h2>
            
            <div class="example">
                <h3>Complete Analysis (Recommended)</h3>
                <span class="method">POST</span> <span class="url">/analyze</span><br>
                Body: <code>{{"text": "Your news article here"}}</code><br>
                <em>Returns: All model results + Verdict Agent final decision</em>
            </div>
            
            <div class="example">
                <h3>Individual Model Predictions</h3>
                <span class="method">POST</span> <span class="url">/predict?model=svm</span><br>
                <span class="method">POST</span> <span class="url">/predict?model=lstm</span><br>
                <span class="method">POST</span> <span class="url">/predict?model=bert</span><br>
                <span class="method">POST</span> <span class="url">/predict-all</span><br>
                <em>Returns: Individual model results only</em>
            </div>
            
            <div class="example">
                <h3>Verdict Agent Only</h3>
                <span class="method">POST</span> <span class="url">/verdict</span><br>
                Body: <code>{{"text": "Article", "model_results": {{"svm": {{"label": "Real", "confidence": 0.95}}}}}}</code><br>
                <em>Returns: Verdict Agent analysis with reasoning</em>
            </div>
            
            <div class="example">
                <h3>System Information</h3>
                <span class="method">GET</span> <span class="url">/system-info</span><br>
                <span class="method">GET</span> <span class="url">/health</span><br>
                <span class="method">GET</span> <span class="url">/stats</span><br>
            </div>
            
            <h3>Example Complete Analysis Response</h3>
            <div class="example">
                <code>
                {{<br>
                &nbsp;&nbsp;"text": "Your news article",<br>
                &nbsp;&nbsp;"model_results": {{<br>
                &nbsp;&nbsp;&nbsp;&nbsp;"SVM": {{"label": "Real", "confidence": 0.9957}},<br>
                &nbsp;&nbsp;&nbsp;&nbsp;"LSTM": {{"label": "Real", "confidence": 0.9848}},<br>
                &nbsp;&nbsp;&nbsp;&nbsp;"BERT": {{"label": "Real", "confidence": 0.9938}}<br>
                &nbsp;&nbsp;}},<br>
                &nbsp;&nbsp;"verdict": {{<br>
                &nbsp;&nbsp;&nbsp;&nbsp;"verdict": "real",<br>
                &nbsp;&nbsp;&nbsp;&nbsp;"confidence": 0.95,<br>
                &nbsp;&nbsp;&nbsp;&nbsp;"reasoning": "Strong agreement across all models...",<br>
                &nbsp;&nbsp;&nbsp;&nbsp;"evidence": [...],<br>
                &nbsp;&nbsp;&nbsp;&nbsp;"explainability": {{...}}<br>
                &nbsp;&nbsp;}}<br>
                }}
                </code>
            </div>
            
            <h3>Multi-Agent System Benefits</h3>
            <ul>
                <li><strong>Diversity:</strong> Multiple AI approaches for robust detection</li>
                <li><strong>Explainability:</strong> Clear reasoning for every decision</li>
                <li><strong>Security:</strong> Input validation and audit trails</li>
                <li><strong>Scalability:</strong> Modular agent architecture</li>
                <li><strong>Transparency:</strong> Full decision-making process visible</li>
            </ul>
        </div>
    </body>
    </html>
    """

@app.route("/analyze", methods=["POST"])
def analyze_complete():
    """
    Complete analysis using all agents
    This is the main endpoint that combines all models with Verdict Agent
    """
    try:
        data = request.json
        text = data.get("text", "").strip()
        
        if not text:
            return jsonify({"error": "Text is required"}), 400
        
        start_time = time.time()
        
        # Get predictions from all models
        model_results = model_predictor.predict_all(text)
        
        if not model_results:
            return jsonify({"error": "No models available"}), 500
        
        # Convert to ModelResult objects for Verdict Agent
        verdict_input = {}
        for model_name, result in model_results.items():
            if "error" not in result:
                verdict_input[model_name] = ModelResult(
                    model_name=model_name,
                    label=result["label"],
                    confidence=result["confidence"],
                    model_type=model_predictor.models[model_name.lower()]["type"],
                    accuracy=model_predictor.models[model_name.lower()]["accuracy"]
                )
        
        # Get Verdict Agent decision
        verdict_response = verdict_agent.process_verdict(text, verdict_input)
        
        # Calculate total processing time
        total_time = int((time.time() - start_time) * 1000)
        
        # Combine results
        response = {
            "text": text,
            "model_results": model_results,
            "verdict": {
                "verdict": verdict_response.verdict.value,
                "confidence": verdict_response.confidence,
                "confidence_level": verdict_response.confidence_level.value,
                "reasoning": verdict_response.reasoning,
                "evidence": [
                    {
                        "source": ev.source,
                        "content": ev.content,
                        "relevance_score": ev.relevance_score,
                        "citation": ev.citation
                    }
                    for ev in verdict_response.evidence
                ],
                "model_agreement": verdict_response.model_agreement,
                "audit_id": verdict_response.audit_id,
                "timestamp": verdict_response.timestamp,
                "processing_time_ms": verdict_response.processing_time_ms,
                "explainability": verdict_response.explainability
            },
            "system_info": {
                "total_processing_time_ms": total_time,
                "agents_used": ["SVM", "LSTM", "BERT", "Verdict Agent"],
                "system_version": SYSTEM_CONFIG["version"]
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in analyze_complete: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route("/verdict", methods=["POST"])
def verdict_only():
    """
    Verdict Agent only - requires model results as input
    """
    try:
        data = request.json
        text = data.get("text", "").strip()
        model_results_raw = data.get("model_results", {})
        
        if not text:
            return jsonify({"error": "Text is required"}), 400
        
        if not model_results_raw:
            return jsonify({"error": "Model results are required"}), 400
        
        # Convert model results
        model_results = {}
        for model_name, result_data in model_results_raw.items():
            model_results[model_name] = ModelResult(
                model_name=model_name,
                label=result_data.get("label", ""),
                confidence=float(result_data.get("confidence", 0.0)),
                model_type=result_data.get("model_type", "Unknown"),
                accuracy=float(result_data.get("accuracy", 0.0))
            )
        
        # Process with Verdict Agent
        verdict_response = verdict_agent.process_verdict(text, model_results)
        
        # Convert to JSON format
        result = {
            "verdict": verdict_response.verdict.value,
            "confidence": verdict_response.confidence,
            "confidence_level": verdict_response.confidence_level.value,
            "reasoning": verdict_response.reasoning,
            "evidence": [
                {
                    "source": ev.source,
                    "content": ev.content,
                    "relevance_score": ev.relevance_score,
                    "citation": ev.citation
                }
                for ev in verdict_response.evidence
            ],
            "model_agreement": verdict_response.model_agreement,
            "audit_id": verdict_response.audit_id,
            "timestamp": verdict_response.timestamp,
            "processing_time_ms": verdict_response.processing_time_ms,
            "explainability": verdict_response.explainability
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in verdict_only: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

# Include all existing endpoints from app_final.py
@app.route("/predict", methods=["POST"])
def predict():
    """Single model prediction (from original system)"""
    data = request.json
    text = data.get("text", "")
    model_type = request.args.get("model", "svm").lower()
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    if model_type == "svm":
        result = model_predictor.predict_svm(text)
    elif model_type == "lstm":
        result = model_predictor.predict_lstm(text)
    elif model_type == "bert":
        result = model_predictor.predict_bert(text)
    else:
        return jsonify({"error": f"Unknown model: {model_type}. Available: svm, lstm, bert"}), 400
    
    if "error" in result:
        return jsonify(result), 500
    
    return jsonify(result)

@app.route("/predict-all", methods=["POST"])
def predict_all():
    """All models prediction (from original system)"""
    data = request.json
    text = data.get("text", "")
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    results = model_predictor.predict_all(text)
    
    if not results:
        return jsonify({"error": "No models available"}), 500
    
    return jsonify(results)

@app.route("/models", methods=["GET"])
def get_models():
    """Model information (from original system)"""
    return jsonify(model_predictor.get_model_info())

@app.route("/health", methods=["GET"])
def health():
    """System health check"""
    model_info = model_predictor.get_model_info()
    return jsonify({
        "status": "ok",
        "system": SYSTEM_CONFIG["name"],
        "version": SYSTEM_CONFIG["version"],
        "available_models": model_info.get("available_models", []),
        "verdict_agent": {
            "status": "active",
            "llm_client": verdict_agent.llm_client is not None,
            "ir_module": verdict_agent.ir_module is not None,
            "nlp_pipeline": verdict_agent.nlp_pipeline is not None
        }
    })

@app.route("/system-info", methods=["GET"])
def system_info():
    """Get complete system information"""
    return jsonify({
        "system": SYSTEM_CONFIG,
        "models": model_predictor.get_model_info(),
        "verdict_agent": {
            "config": verdict_agent.config,
            "audit_log_size": len(verdict_agent.audit_log)
        }
    })

@app.route("/stats", methods=["GET"])
def get_stats():
    """Get system statistics"""
    model_info = model_predictor.get_model_info()
    
    return jsonify({
        "system": {
            "name": SYSTEM_CONFIG["name"],
            "version": SYSTEM_CONFIG["version"],
            "agents": len(SYSTEM_CONFIG["agents"])
        },
        "models": {
            "available": model_info.get("available_models", []),
            "total": len(model_info.get("available_models", []))
        },
        "verdict_agent": {
            "total_verdicts": len(verdict_agent.audit_log),
            "audit_logging": verdict_agent.config["security"]["enable_audit_logging"]
        }
    })

if __name__ == "__main__":
    print("\n🚀 Starting Multi-Agent Fake News Detection System...")
    print("System Components:")
    print("  ✅ SVM Agent (Traditional ML)")
    print("  ✅ LSTM Agent (Deep Learning)")
    print("  ✅ BERT Agent (Transformer)")
    print("  ✅ Verdict Agent (LLM Reasoning + Explainability)")
    print("\nAvailable endpoints:")
    print("  GET  / - Homepage with documentation")
    print("  GET  /health - Health check")
    print("  GET  /system-info - System information")
    print("  GET  /stats - Statistics")
    print("  POST /analyze - Complete analysis (RECOMMENDED)")
    print("  POST /verdict - Verdict Agent only")
    print("  POST /predict?model=svm - SVM prediction")
    print("  POST /predict?model=lstm - LSTM prediction")
    print("  POST /predict?model=bert - BERT prediction")
    print("  POST /predict-all - All models prediction")
    print("  GET  /models - Model information")
    print("\nStarting server on http://localhost:5000")
    app.run(debug=False, use_reloader=False)
