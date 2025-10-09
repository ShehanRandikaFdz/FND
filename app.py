"""
Fake News Detection App - Streamlit + Hugging Face Deployment
Multi-page application with model comparison, statistics, and comprehensive analysis
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Import our custom modules
from utils.model_loader import load_models, get_model_info
from utils.predictor import UnifiedPredictor

# Page configuration
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .model-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #f9f9f9;
    }
    .prediction-result {
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .fake-result {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .true-result {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
    }
    .uncertain-result {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

# Load models once
@st.cache_resource
def initialize_models():
    """Initialize models and predictor"""
    model_loader = load_models()
    if model_loader:
        predictor = UnifiedPredictor(model_loader)
        return model_loader, predictor
    return None, None

def main():
    """Main application"""
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Fake News Detection System</h1>', unsafe_allow_html=True)
    
    # Initialize models
    model_loader, predictor = initialize_models()
    
    if model_loader is None:
        st.error("❌ Failed to load models. Please check that model files are present in the 'models' directory.")
        st.stop()
    
    st.session_state.models_loaded = True
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Main Analysis", "📊 Model Comparison", "📈 Statistics & Insights", "ℹ️ About"]
    )
    
    # Page routing
    if page == "🏠 Main Analysis":
        main_analysis_page(predictor)
    elif page == "📊 Model Comparison":
        model_comparison_page()
    elif page == "📈 Statistics & Insights":
        statistics_page()
    elif page == "ℹ️ About":
        about_page()

def main_analysis_page(predictor):
    """Main prediction interface"""
    
    st.header("📝 Text Analysis")
    st.markdown("Enter the news text you want to analyze for authenticity.")
    
    # Input form
    with st.form("analysis_form"):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            text = st.text_area(
                "News Text:",
                placeholder="Enter the news text here...",
                height=150,
                help="Minimum 10 characters, maximum 5000 characters"
            )
        
        with col2:
            title = st.text_input("Title (optional):", placeholder="News title")
            source = st.text_input("Source (optional):", placeholder="News source")
        
        analyze_button = st.form_submit_button("🔍 Analyze News", type="primary")
    
    if analyze_button:
        if not text or len(text.strip()) < 10:
            st.error("Please enter at least 10 characters of text.")
        else:
            with st.spinner("Analyzing text..."):
                # Perform analysis
                result = predictor.analyze_text(text, title, source)
                
                # Store in history
                result['timestamp'] = datetime.now().isoformat()
                st.session_state.analysis_history.append(result)
                
                # Display results
                display_analysis_results(result)

def display_analysis_results(result):
    """Display comprehensive analysis results"""
    
    ensemble_result = result.get('ensemble_result', {})
    prediction = ensemble_result.get('overall_prediction', 'UNKNOWN')
    confidence = ensemble_result.get('overall_confidence', 0)
    
    # Overall verdict
    st.header("🎯 Analysis Results")
    
    # Determine styling based on prediction
    if prediction == 'FAKE':
        result_class = "fake-result"
        verdict_emoji = "❌"
        verdict_color = "#f44336"
    elif prediction == 'TRUE':
        result_class = "true-result"
        verdict_emoji = "✅"
        verdict_color = "#4caf50"
    else:
        result_class = "uncertain-result"
        verdict_emoji = "⚠️"
        verdict_color = "#ff9800"
    
    # Display overall verdict
    st.markdown(f"""
    <div class="prediction-result {result_class}">
        <h2>{verdict_emoji} Overall Verdict: {prediction}</h2>
        <h3>Confidence: {confidence:.1f}%</h3>
        <p>{result.get('analysis_summary', '')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Confidence meter
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Level"},
        delta = {'reference': 80},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': verdict_color},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # Individual model predictions
    st.header("🤖 Individual Model Predictions")
    
    predictions = result.get('individual_predictions', [])
    for pred in predictions:
        if pred.get('prediction') != 'ERROR':
            with st.expander(f"{pred['model_name']} Model Results"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Prediction", pred['prediction'])
                
                with col2:
                    st.metric("Confidence", f"{pred['confidence']:.1f}%")
                
                with col3:
                    st.metric("Fake Probability", f"{pred.get('probability_fake', 0):.1f}%")
    
    # Risk factors
    risk_factors = result.get('risk_factors', [])
    if risk_factors:
        st.header("⚠️ Risk Factors Identified")
        for factor in risk_factors:
            st.warning(f"• {factor}")
    
    # Credibility analysis
    credibility_analysis = result.get('credibility_analysis', {})
    if credibility_analysis.get('available'):
        st.header("📊 Credibility Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Credibility Score", f"{credibility_analysis.get('credibility_score', 0):.2f}")
        with col2:
            st.metric("Uncertainty", f"{credibility_analysis.get('uncertainty', 0):.2f}")
    
    # Verdict explanation
    verdict = result.get('verdict', {})
    if verdict.get('available'):
        st.header("🎯 Final Verdict")
        st.info(verdict.get('explanation', 'No explanation available'))

def model_comparison_page():
    """Model comparison and performance metrics"""
    
    st.header("📊 Model Comparison")
    
    model_info = get_model_info()
    
    # Performance metrics table
    st.subheader("Performance Metrics")
    
    metrics_data = []
    for model_key, info in model_info.items():
        metrics_data.append({
            'Model': info['name'],
            'Accuracy (%)': info['accuracy'],
            'Type': info['description'],
            'Strength': info['strength']
        })
    
    df = pd.DataFrame(metrics_data)
    st.dataframe(df, use_container_width=True)
    
    # Model comparison visualization
    st.subheader("Model Accuracy Comparison")
    
    fig = px.bar(
        df,
        x='Model',
        y='Accuracy (%)',
        color='Accuracy (%)',
        color_continuous_scale='RdYlGn',
        title="Model Accuracy Comparison"
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Model characteristics
    st.subheader("Model Characteristics")
    
    for model_key, info in model_info.items():
        with st.expander(f"{info['name']} Details"):
            st.write(f"**Description:** {info['description']}")
            st.write(f"**Key Strength:** {info['strength']}")
            st.write(f"**Accuracy:** {info['accuracy']}%")
            
            # Add model-specific information
            if model_key == 'svm':
                st.write("**Technical Details:**")
                st.write("- Uses TF-IDF vectorization")
                st.write("- Linear kernel support vector machine")
                st.write("- Excellent for structured text features")
            
            elif model_key == 'lstm':
                st.write("**Technical Details:**")
                st.write("- Recurrent neural network architecture")
                st.write("- Captures sequential patterns in text")
                st.write("- Good for understanding context flow")
            
            elif model_key == 'bert':
                st.write("**Technical Details:**")
                st.write("- Transformer-based architecture")
                st.write("- Bidirectional attention mechanism")
                st.write("- State-of-the-art for text understanding")

def statistics_page():
    """Statistics and insights from analysis history"""
    
    st.header("📈 Statistics & Insights")
    
    if not st.session_state.analysis_history:
        st.info("No analysis history available. Perform some analyses to see statistics here.")
        return
    
    history = st.session_state.analysis_history
    
    # Basic statistics
    st.subheader("Session Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Analyses", len(history))
    
    # Count predictions
    fake_count = sum(1 for h in history if h.get('ensemble_result', {}).get('overall_prediction') == 'FAKE')
    true_count = sum(1 for h in history if h.get('ensemble_result', {}).get('overall_prediction') == 'TRUE')
    
    with col2:
        st.metric("Fake News Detected", fake_count)
    
    with col3:
        st.metric("True News Verified", true_count)
    
    with col4:
        avg_confidence = np.mean([h.get('ensemble_result', {}).get('overall_confidence', 0) for h in history])
        st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
    
    # Prediction distribution
    st.subheader("Prediction Distribution")
    
    prediction_counts = {'FAKE': fake_count, 'TRUE': true_count}
    
    fig = px.pie(
        values=list(prediction_counts.values()),
        names=list(prediction_counts.keys()),
        title="Distribution of Predictions",
        color_discrete_map={'FAKE': '#f44336', 'TRUE': '#4caf50'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Confidence distribution
    st.subheader("Confidence Distribution")
    
    confidences = [h.get('ensemble_result', {}).get('overall_confidence', 0) for h in history]
    
    fig = px.histogram(
        x=confidences,
        nbins=20,
        title="Distribution of Confidence Scores",
        labels={'x': 'Confidence (%)', 'y': 'Count'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Model agreement analysis
    st.subheader("Model Agreement Analysis")
    
    agreement_data = []
    for h in history:
        predictions = h.get('individual_predictions', [])
        if len(predictions) >= 2:
            preds = [p.get('prediction') for p in predictions if p.get('prediction') != 'ERROR']
            if len(set(preds)) == 1:  # All models agree
                agreement_data.append('Full Agreement')
            elif len(set(preds)) == 2:  # Some disagreement
                agreement_data.append('Partial Disagreement')
            else:
                agreement_data.append('Major Disagreement')
    
    if agreement_data:
        agreement_counts = pd.Series(agreement_data).value_counts()
        
        fig = px.bar(
            x=agreement_counts.index,
            y=agreement_counts.values,
            title="Model Agreement Analysis",
            labels={'x': 'Agreement Level', 'y': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk factors analysis
    st.subheader("Most Common Risk Factors")
    
    all_risk_factors = []
    for h in history:
        all_risk_factors.extend(h.get('risk_factors', []))
    
    if all_risk_factors:
        risk_factor_counts = pd.Series(all_risk_factors).value_counts()
        
        fig = px.bar(
            x=risk_factor_counts.values,
            y=risk_factor_counts.index,
            orientation='h',
            title="Most Common Risk Factors",
            labels={'x': 'Frequency', 'y': 'Risk Factor'}
        )
        st.plotly_chart(fig, use_container_width=True)

def about_page():
    """About page with project information"""
    
    st.header("ℹ️ About This Project")
    
    st.markdown("""
    ## 🔍 Fake News Detection System
    
    This application uses advanced machine learning models to analyze and detect fake news with high accuracy.
    
    ### 🎯 Features
    
    - **Multi-Model Analysis**: Combines SVM, LSTM, and BERT models for robust detection
    - **Real-time Analysis**: Get instant results with detailed explanations
    - **Credibility Assessment**: Advanced credibility analysis with risk factor identification
    - **Model Comparison**: Compare performance of different models
    - **Statistical Insights**: Track analysis patterns and model performance
    
    ### 🤖 Model Architecture
    
    #### 1. Support Vector Machine (SVM)
    - **Accuracy**: 99.5%
    - **Type**: Traditional machine learning with TF-IDF features
    - **Strength**: Excellent performance on structured text features
    
    #### 2. Long Short-Term Memory (LSTM)
    - **Accuracy**: 87.0%
    - **Type**: Deep learning for sequential data
    - **Strength**: Captures temporal patterns and context flow
    
    #### 3. DistilBERT
    - **Accuracy**: 75.0%
    - **Type**: Transformer model with attention mechanism
    - **Strength**: State-of-the-art text understanding and semantics
    
    ### 🔧 Technical Implementation
    
    - **Frontend**: Streamlit for interactive web interface
    - **Backend**: Python with TensorFlow, PyTorch, and scikit-learn
    - **Deployment**: Hugging Face Spaces for easy access
    - **Memory Optimization**: BERT model uses half-precision for efficiency
    
    ### ⚠️ Important Limitations
    
    - This tool is for educational and research purposes
    - Results should not be the sole basis for important decisions
    - Models may have biases based on training data
    - Always verify information through multiple reliable sources
    
    ### 📚 Responsible Use
    
    - Use as a supplementary tool for fact-checking
    - Combine with human judgment and additional verification
    - Be aware of potential biases in automated systems
    - Respect privacy and ethical considerations
    
    ### 🚀 Deployment
    
    This application is deployed on Hugging Face Spaces, making it accessible to users worldwide.
    
    ### 📞 Support
    
    For questions or issues, please refer to the GitHub repository or Hugging Face Space.
    
    ---
    
    **Built with ❤️ using Streamlit, TensorFlow, PyTorch, and Hugging Face**
    """)

if __name__ == "__main__":
    main()