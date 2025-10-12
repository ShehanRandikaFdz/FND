"""
Unified Fake News Detection & Verification System
Combines ML prediction with automatic online news verification
"""

import streamlit as st
import os
import sys
import time
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Quick compatibility check
try:
    from utils.compatibility import fix_numpy_compatibility
    fix_numpy_compatibility()
except ImportError:
    pass

# Page configuration
st.set_page_config(
    page_title="Fake News Detection & Verification System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(120deg, #2196F3, #00BCD4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }
    .prediction-result {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .fake-result {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .true-result {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
    .verified-result {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
    }
    .confidence-high {
        color: #4caf50;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ff9800;
        font-weight: bold;
    }
    .confidence-low {
        color: #f44336;
        font-weight: bold;
    }
    .model-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #007bff;
    }
    .verification-card {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #17a2b8;
    }
    .article-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border: 1px solid #dee2e6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
if 'loading_time' not in st.session_state:
    st.session_state.loading_time = None
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None

@st.cache_resource
def load_system():
    """Load the ML system components"""
    try:
        from utils.model_loader import ModelLoader
        from utils.predictor import UnifiedPredictor
        from utils.text_preprocessor import TextPreprocessor
        
        model_loader = ModelLoader()
        
        # Load all models
        model_loader.load_svm_model()
        model_loader.load_lstm_model()
        model_loader.load_bert_model()
        
        predictor = UnifiedPredictor(model_loader)
        preprocessor = TextPreprocessor()
        
        return model_loader, predictor, preprocessor, True
    except Exception as e:
        st.error(f"Error loading system: {e}")
        return None, None, None, False

def display_ml_predictions(ml_results):
    """Display ML prediction results"""
    st.subheader("🤖 ML Model Predictions")
    
    if "error" in ml_results:
        st.error(f"Error: {ml_results['error']}")
        return
    
    final_prediction = ml_results.get("final_prediction", "UNKNOWN")
    confidence = ml_results.get("confidence", 0)
    individual_results = ml_results.get("individual_results", {})
    votes = ml_results.get("votes", {"FAKE": 0, "TRUE": 0})
    
    # Display individual model results
    if individual_results:
        for model_name, model_result in individual_results.items():
            pred = model_result.get("prediction", "UNKNOWN")
            conf = model_result.get("confidence", 0)
            
            # Model-specific styling
            if pred == "FAKE":
                model_icon = "🔴"
                model_color = "#ffcdd2"
            else:
                model_icon = "🟢"
                model_color = "#c8e6c9"
            
            with st.container():
                st.markdown(f"""
                <div class="model-card">
                    <strong>{model_icon} {model_name.upper()}</strong>: {pred} ({conf:.1f}%)
                </div>
                """, unsafe_allow_html=True)
    
    # Display voting breakdown
    col1, col2 = st.columns(2)
    with col1:
        st.metric("FAKE Votes", votes["FAKE"])
    with col2:
        st.metric("TRUE Votes", votes["TRUE"])
    
    # Display final ML result
    if final_prediction == "FAKE":
        result_class = "fake-result"
        icon = "🚨"
    else:
        result_class = "true-result"
        icon = "✅"
    
    st.markdown(f"""
    <div class="prediction-result {result_class}">
        <h4>{icon} ML Ensemble Result: {final_prediction}</h4>
        <p class="confidence-high">Confidence: {confidence:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)

def display_verification_results(verification_result):
    """Display online verification results"""
    st.subheader("📰 Online Verification")
    
    if not verification_result:
        st.info("ℹ️ Verification not performed")
        return
    
    if verification_result.get('found_online', False):
        best_match = verification_result.get('best_match', {})
        similarity_score = verification_result.get('similarity_score', 0)
        all_matches = verification_result.get('all_matches', [])
        
        # Display verification summary
        if similarity_score > 0.8:
            status_icon = "✅"
            status_text = "Highly Verified"
            status_color = "#4caf50"
        elif similarity_score > 0.6:
            status_icon = "✅"
            status_text = "Likely Verified"
            status_color = "#8bc34a"
        elif similarity_score > 0.4:
            status_icon = "⚠️"
            status_text = "Partially Verified"
            status_color = "#ff9800"
        else:
            status_icon = "❓"
            status_text = "Weak Match"
            status_color = "#ff5722"
        
        st.markdown(f"""
        <div class="verification-card">
            <h4>{status_icon} {status_text}</h4>
            <p><strong>Similarity Score:</strong> {similarity_score:.1%}</p>
            <p><strong>Articles Found:</strong> {len(all_matches)}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display best match
        if best_match:
            st.markdown("**📄 Best Match:**")
            source_name = best_match.get('source', {}).get('name', 'Unknown Source')
            title = best_match.get('title', 'No title')
            description = best_match.get('description', 'No description')
            url = best_match.get('url', '#')
            published_at = best_match.get('publishedAt', 'Unknown date')
            
            # Format date
            try:
                if published_at and published_at != 'Unknown date':
                    pub_date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                    formatted_date = pub_date.strftime('%Y-%m-%d %H:%M')
                else:
                    formatted_date = published_at
            except:
                formatted_date = published_at
            
            with st.container():
                st.markdown(f"""
                <div class="article-card">
                    <h5>{title}</h5>
                    <p><strong>Source:</strong> {source_name}</p>
                    <p><strong>Published:</strong> {formatted_date}</p>
                    <p><strong>Similarity:</strong> {similarity_score:.1%}</p>
                    <p>{description}</p>
                    <a href="{url}" target="_blank">Read Full Article →</a>
                </div>
                """, unsafe_allow_html=True)
        
        # Display other matches
        if len(all_matches) > 1:
            with st.expander(f"📋 View All Matches ({len(all_matches)})"):
                for i, match in enumerate(all_matches[1:], 2):
                    source_name = match.get('source', {}).get('name', 'Unknown')
                    title = match.get('title', 'No title')
                    similarity = match.get('similarity_score', 0)
                    url = match.get('url', '#')
                    
                    st.markdown(f"""
                    **{i}.** [{title}]({url})  
                    *{source_name}* - Similarity: {similarity:.1%}
                    """)
    else:
        error_msg = verification_result.get('error', 'No matching articles found')
        st.markdown(f"""
        <div class="verification-card">
            <h4>❌ Not Found Online</h4>
            <p>{error_msg}</p>
        </div>
        """, unsafe_allow_html=True)

def display_final_assessment(final_assessment, ml_results, verification_result):
    """Display the final credibility assessment"""
    st.markdown("---")
    st.subheader("🎯 Final Assessment")
    
    # Determine assessment styling
    if 'VERIFIED_TRUE' in final_assessment:
        assessment_class = "verified-result"
        assessment_icon = "✅"
        assessment_color = "#2196f3"
    elif 'LIKELY_TRUE' in final_assessment:
        assessment_class = "true-result"
        assessment_icon = "✅"
        assessment_color = "#4caf50"
    elif 'LIKELY_FAKE' in final_assessment:
        assessment_class = "fake-result"
        assessment_icon = "🚨"
        assessment_color = "#f44336"
    elif 'CONFLICTING' in final_assessment:
        assessment_class = "prediction-result"
        assessment_icon = "⚠️"
        assessment_color = "#ff9800"
    else:
        assessment_class = "prediction-result"
        assessment_icon = "❓"
        assessment_color = "#666"
    
    # Create assessment explanation
    explanation = get_assessment_explanation(final_assessment, ml_results, verification_result)
    
    st.markdown(f"""
    <div class="{assessment_class}">
        <h3>{assessment_icon} {final_assessment.replace('_', ' ').title()}</h3>
        <p>{explanation}</p>
    </div>
    """, unsafe_allow_html=True)

def get_assessment_explanation(assessment, ml_results, verification_result):
    """Generate explanation for the final assessment"""
    explanations = {
        'VERIFIED_TRUE': "This content appears to be TRUE and has been verified against online sources with high confidence.",
        'LIKELY_TRUE_VERIFIED': "This content appears to be TRUE and has been partially verified against online sources.",
        'LIKELY_TRUE_NOT_VERIFIED': "This content appears to be TRUE based on ML analysis, but no matching online sources were found.",
        'LIKELY_FAKE_NOT_FOUND': "This content appears to be FAKE based on ML analysis and was not found in reputable online sources.",
        'CONFLICTING_HIGH_SIMILARITY': "⚠️ CONFLICT: ML models suggest FAKE, but content was found in reputable sources with high similarity.",
        'CONFLICTING_MEDIUM_SIMILARITY': "⚠️ CONFLICT: ML models suggest FAKE, but similar content was found in online sources.",
        'PARTIALLY_VERIFIED': "Content was found online but with low similarity - requires further investigation.",
        'WEAK_VERIFICATION': "Content was found online but similarity is very low - likely not the same story.",
        'UNKNOWN_NOT_VERIFIED': "Unable to determine credibility - no clear indicators found."
    }
    
    return explanations.get(assessment, f"Assessment: {assessment}")

def main():
    """Main application interface"""
    st.markdown('<h1 class="main-header">🔍 Fake News Detection & Verification System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced ML Analysis + Online Source Verification</p>', unsafe_allow_html=True)
    
    # Load system components
    if st.session_state.analyzer is None:
        with st.spinner("⚡ Loading ML models and verification system..."):
            start_time = time.time()
            model_loader, predictor, preprocessor, success = load_system()
            if success:
                st.session_state.analyzer = model_loader
                st.session_state.predictor = predictor
                st.session_state.preprocessor = preprocessor
                st.session_state.loading_time = time.time() - start_time
                st.success(f"✅ System loaded in {st.session_state.loading_time:.1f}s - Ready for analysis!")
            else:
                st.error("❌ Failed to load system components")
                return
    
    # Main input section
    st.markdown("### 📝 Enter News Text for Analysis")
    
    # Text input
    text_input = st.text_area(
        "Paste or type the news text you want to analyze:",
        height=150,
        placeholder="Enter news article text here...",
        help="Minimum 10 characters. The system will analyze this text using ML models and optionally verify it against online sources."
    )
    
    # Options
    col1, col2 = st.columns([1, 1])
    
    with col1:
        enable_verification = st.checkbox(
            "🔍 Enable Online Verification", 
            value=True,
            help="Search for this text in online news sources to verify authenticity"
        )
    
    with col2:
        show_preprocessing = st.checkbox(
            "🔧 Show Text Preprocessing Steps",
            value=False,
            help="Display step-by-step text preprocessing"
        )
    
    # Analyze button
    if st.button("🚀 Analyze Text", type="primary", use_container_width=True):
        if not text_input or len(text_input.strip()) < 10:
            st.warning("⚠️ Please enter at least 10 characters of text to analyze.")
            return
        
        # Show preprocessing if requested
        if show_preprocessing:
            st.markdown("---")
            st.subheader("🔧 Text Preprocessing")
            
            preprocessing_result = st.session_state.preprocessor.preprocess(text_input, show_steps=True)
            stats = st.session_state.preprocessor.get_preprocessing_stats(text_input, preprocessing_result['processed'])
            
            # Display preprocessing steps
            if preprocessing_result['steps']:
                html_steps = st.session_state.preprocessor.display_preprocessing_steps(preprocessing_result['steps'])
                st.markdown(html_steps, unsafe_allow_html=True)
            
            # Display stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Length", f"{stats['original_length']} chars")
            with col2:
                st.metric("Processed Length", f"{stats['processed_length']} chars")
            with col3:
                st.metric("Reduction", f"{stats['reduction_percentage']:.1f}%")
        
        # Perform analysis
        with st.spinner("🔍 Analyzing text..."):
            try:
                # Get unified prediction and verification
                results = st.session_state.predictor.predict_and_verify(
                    text_input, 
                    enable_verification=enable_verification
                )
                
                ml_results = results['ml_prediction']
                verification_result = results['verification'] if enable_verification else None
                final_assessment = results['final_assessment']
                
                # Display results in two columns
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    display_ml_predictions(ml_results)
                
                with col2:
                    if enable_verification:
                        display_verification_results(verification_result)
                    else:
                        st.info("ℹ️ Online verification disabled")
                
                # Display final assessment
                display_final_assessment(final_assessment, ml_results, verification_result)
                
                # Store results in session state for potential export
                st.session_state.last_results = results
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {e}")
    
    # Real-time monitoring section
    st.markdown("---")
    with st.expander("📰 Real-Time News Monitoring", expanded=False):
        st.markdown("### 🌐 Live News Feed Analysis")
        st.info("💡 This feature allows you to monitor live news feeds and analyze their credibility in real-time.")
        
        # Check if NewsAPI is configured
        try:
            from config import config
            if hasattr(config, 'validate') and config.validate():
                st.success("✅ NewsAPI configured - Real-time monitoring available")
                
                # Simple news fetching interface
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    country = st.selectbox(
                        "Select Country:",
                        ['us', 'gb', 'ca', 'au', 'de', 'fr', 'in', 'jp'],
                        index=0
                    )
                
                with col2:
                    category = st.selectbox(
                        "Select Category:",
                        ['general', 'business', 'technology', 'science', 'health', 'sports', 'entertainment'],
                        index=0
                    )
                
                if st.button("📡 Fetch Latest News", type="secondary"):
                    with st.spinner("Fetching latest news..."):
                        try:
                            from news_fetcher import NewsFetcher
                            fetcher = NewsFetcher()
                            articles = fetcher.fetch_and_analyze(
                                country=country,
                                category=category,
                                page_size=10
                            )
                            
                            if articles:
                                st.success(f"✅ Fetched {len(articles)} articles")
                                
                                for i, article in enumerate(articles[:5], 1):
                                    credibility = article.get('credibility_score', 0)
                                    prediction = article.get('prediction', 'UNKNOWN')
                                    
                                    if prediction == 'FAKE':
                                        status_icon = "🚨"
                                        status_color = "#ffebee"
                                    elif prediction == 'TRUE':
                                        status_icon = "✅"
                                        status_color = "#e8f5e9"
                                    else:
                                        status_icon = "❓"
                                        status_color = "#f5f5f5"
                                    
                                    st.markdown(f"""
                                    <div style="background-color: {status_color}; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0;">
                                        <h5>{status_icon} {article.get('title', 'No title')}</h5>
                                        <p><strong>Source:</strong> {article.get('source', {}).get('name', 'Unknown')}</p>
                                        <p><strong>Prediction:</strong> {prediction} (Confidence: {credibility:.1%})</p>
                                        <p>{article.get('description', 'No description')}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.warning("No articles fetched")
                                
                        except ImportError:
                            st.warning("⚠️ News fetcher not available. Please ensure all dependencies are installed.")
                        except Exception as e:
                            st.error(f"Failed to fetch news: {e}")
            else:
                st.warning("⚠️ NewsAPI not configured. Please set NEWSAPI_KEY in your environment.")
                st.code("""
# To enable real-time monitoring:
# 1. Get API key from https://newsapi.org/register
# 2. Create .env file with:
# NEWSAPI_KEY=your_api_key_here
                """)
                
        except ImportError:
            st.warning("⚠️ Real-time monitoring components not available")

if __name__ == "__main__":
    main()