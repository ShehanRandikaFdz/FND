"""
Real-Time Fake News Detection Dashboard
Live news monitoring with credibility analysis
"""
import streamlit as st
import time
from datetime import datetime
from typing import List, Dict
import pandas as pd

from news_fetcher import NewsFetcher
from config import config

# Page configuration
st.set_page_config(
    page_title="Real-Time Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(120deg, #2196F3, #00BCD4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .fake-news {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    .suspicious-news {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    .credible-news {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
    }
    .article-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .article-meta {
        color: #666;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    .score-badge {
        display: inline-block;
        padding: 0.3rem 0.6rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .score-fake {
        background-color: #f44336;
        color: white;
    }
    .score-suspicious {
        background-color: #ff9800;
        color: white;
    }
    .score-credible {
        background-color: #4caf50;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_fetcher():
    """Initialize news fetcher (cached)"""
    return NewsFetcher()


def display_article(article: Dict, index: int):
    """Display single article with credibility analysis"""
    score = article['credibility_score']
    
    # Determine styling based on credibility score
    if score < config.FAKE_THRESHOLD:
        css_class = "fake-news"
        status_emoji = "🚨"
        status_text = "LIKELY FAKE"
        badge_class = "score-fake"
    elif score < config.SUSPICIOUS_THRESHOLD:
        css_class = "suspicious-news"
        status_emoji = "⚠️"
        status_text = "SUSPICIOUS"
        badge_class = "score-suspicious"
    else:
        css_class = "credible-news"
        status_emoji = "✅"
        status_text = "CREDIBLE"
        badge_class = "score-credible"
    
    # Article container
    st.markdown(f"""
    <div class="{css_class}">
        <div class="article-title">{status_emoji} {article['title']}</div>
        <div class="article-meta">
            <strong>Source:</strong> {article['source']} | 
            <strong>Author:</strong> {article.get('author', 'Unknown')} | 
            <strong>Published:</strong> {article.get('published_at', 'Unknown')[:10]}
        </div>
        <p style="margin: 0.5rem 0;">{article.get('content', '')[:200]}...</p>
        <div style="margin-top: 0.5rem;">
            <span class="score-badge {badge_class}">
                {status_text} • Score: {score:.2f} • Confidence: {article['confidence']:.2f}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Expandable details
    with st.expander("📊 View Detailed Analysis"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Individual Model Predictions:**")
            for model, pred in article['individual_predictions'].items():
                label = pred['label']
                conf = pred['confidence']
                emoji = "✅" if label == "Real" else "🚨"
                st.write(f"{emoji} **{model.upper()}**: {label} ({conf:.2%})")
        
        with col2:
            st.markdown("**Article Information:**")
            st.write(f"**URL:** {article['url']}")
            st.write(f"**Analyzed At:** {article.get('analyzed_at', 'Unknown')[:19]}")
            
            if article.get('image_url'):
                st.image(article['image_url'], use_container_width=True)
    
    st.markdown("---")


def display_statistics(articles: List[Dict], fetcher: NewsFetcher):
    """Display statistics dashboard"""
    if not articles:
        st.info("No articles to display statistics")
        return
    
    stats = fetcher.get_statistics(articles)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="color: #2196F3; margin: 0;">{stats['total']}</h2>
            <p style="margin: 0; color: #666;">Total Articles</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="color: #f44336; margin: 0;">{stats['fake']}</h2>
            <p style="margin: 0; color: #666;">🚨 Likely Fake ({stats['fake_percentage']:.1f}%)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="color: #ff9800; margin: 0;">{stats['suspicious']}</h2>
            <p style="margin: 0; color: #666;">⚠️ Suspicious ({stats['suspicious_percentage']:.1f}%)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="color: #4caf50; margin: 0;">{stats['credible']}</h2>
            <p style="margin: 0; color: #666;">✅ Credible ({stats['credible_percentage']:.1f}%)</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)


def main():
    """Main application"""
    
    # Header
    st.markdown('<div class="main-header">📰 Real-Time Fake News Detector</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Live news monitoring with AI-powered credibility analysis</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    # Check configuration
    if not config.NEWSAPI_KEY:
        st.error("❌ NewsAPI key not found!")
        st.info("""
        **Setup Steps:**
        1. Get a free API key from [NewsAPI.org](https://newsapi.org/register)
        2. Copy `.env.example` to `.env`
        3. Add your API key to the `.env` file
        4. Restart the application
        """)
        st.stop()
    
    # Initialize fetcher
    try:
        fetcher = get_fetcher()
    except Exception as e:
        st.error(f"❌ Failed to initialize: {e}")
        st.stop()
    
    # Fetch mode selection
    fetch_mode = st.sidebar.radio(
        "📡 Fetch Mode",
        ["Top Headlines", "Search by Topic", "By Category"]
    )
    
    # Mode-specific settings
    if fetch_mode == "Top Headlines":
        country = st.sidebar.selectbox(
            "🌍 Country",
            options=["us", "gb", "ca", "au", "in", "de", "fr", "it", "es"],
            format_func=lambda x: {
                "us": "🇺🇸 United States",
                "gb": "🇬🇧 United Kingdom",
                "ca": "🇨🇦 Canada",
                "au": "🇦🇺 Australia",
                "in": "🇮🇳 India",
                "de": "🇩🇪 Germany",
                "fr": "🇫🇷 France",
                "it": "🇮🇹 Italy",
                "es": "🇪🇸 Spain"
            }[x]
        )
        
        page_size = st.sidebar.slider("📄 Articles to Fetch", 5, 50, 20)
        
        fetch_params = {
            'country': country,
            'page_size': page_size
        }
    
    elif fetch_mode == "Search by Topic":
        search_query = st.sidebar.text_input("🔍 Search Query", "technology")
        page_size = st.sidebar.slider("📄 Articles to Fetch", 5, 50, 20)
        
        fetch_params = {
            'query': search_query,
            'page_size': page_size
        }
    
    else:  # By Category
        category = st.sidebar.selectbox(
            "📂 Category",
            options=["general", "business", "entertainment", "health", "science", "sports", "technology"]
        )
        country = st.sidebar.selectbox(
            "🌍 Country",
            options=["us", "gb", "ca", "au", "in"],
            format_func=lambda x: {
                "us": "🇺🇸 United States",
                "gb": "🇬🇧 United Kingdom",
                "ca": "🇨🇦 Canada",
                "au": "🇦🇺 Australia",
                "in": "🇮🇳 India"
            }[x]
        )
        page_size = st.sidebar.slider("📄 Articles to Fetch", 5, 50, 20)
        
        fetch_params = {
            'country': country,
            'category': category,
            'page_size': page_size
        }
    
    # Filter options
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Filter Results")
    
    show_fake = st.sidebar.checkbox("🚨 Show Likely Fake", value=True)
    show_suspicious = st.sidebar.checkbox("⚠️ Show Suspicious", value=True)
    show_credible = st.sidebar.checkbox("✅ Show Credible", value=True)
    
    # Fetch button
    st.sidebar.markdown("---")
    fetch_button = st.sidebar.button("🔄 Fetch & Analyze News", type="primary", use_container_width=True)
    
    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (5 min)")
    
    # Main content area
    if 'articles' not in st.session_state:
        st.session_state.articles = []
        st.session_state.last_fetch = None
    
    # Fetch news
    if fetch_button or (auto_refresh and (
        st.session_state.last_fetch is None or 
        (time.time() - st.session_state.last_fetch) > 300
    )):
        with st.spinner("🔄 Fetching and analyzing news... This may take a minute..."):
            try:
                if fetch_mode == "Search by Topic":
                    articles = fetcher.search_and_analyze(**fetch_params)
                else:
                    articles = fetcher.fetch_and_analyze(**fetch_params)
                
                st.session_state.articles = articles
                st.session_state.last_fetch = time.time()
                
                if articles:
                    st.success(f"✅ Successfully analyzed {len(articles)} articles!")
                else:
                    st.warning("⚠️ No articles found. Try different settings.")
            except Exception as e:
                st.error(f"❌ Error fetching news: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    # Display results
    articles = st.session_state.articles
    
    if articles:
        # Show last fetch time
        if st.session_state.last_fetch:
            last_fetch_time = datetime.fromtimestamp(st.session_state.last_fetch)
            st.info(f"📅 Last updated: {last_fetch_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Statistics
        display_statistics(articles, fetcher)
        
        # Filter articles
        filtered_articles = []
        for article in articles:
            score = article['credibility_score']
            if score < config.FAKE_THRESHOLD and show_fake:
                filtered_articles.append(article)
            elif config.FAKE_THRESHOLD <= score < config.SUSPICIOUS_THRESHOLD and show_suspicious:
                filtered_articles.append(article)
            elif score >= config.SUSPICIOUS_THRESHOLD and show_credible:
                filtered_articles.append(article)
        
        # Sort options
        sort_by = st.selectbox(
            "📊 Sort by",
            options=["credibility_score_asc", "credibility_score_desc", "published_at", "source"],
            format_func=lambda x: {
                "credibility_score_asc": "Credibility Score (Low to High)",
                "credibility_score_desc": "Credibility Score (High to Low)",
                "published_at": "Published Date",
                "source": "Source Name"
            }[x]
        )
        
        # Sort articles
        if sort_by == "credibility_score_asc":
            filtered_articles.sort(key=lambda x: x['credibility_score'])
        elif sort_by == "credibility_score_desc":
            filtered_articles.sort(key=lambda x: x['credibility_score'], reverse=True)
        elif sort_by == "published_at":
            filtered_articles.sort(key=lambda x: x.get('published_at', ''), reverse=True)
        else:  # source
            filtered_articles.sort(key=lambda x: x['source'])
        
        # Display articles
        st.markdown(f"### 📰 Showing {len(filtered_articles)} articles")
        st.markdown("---")
        
        for i, article in enumerate(filtered_articles):
            display_article(article, i)
    
    else:
        # Welcome message
        st.info("""
        👋 **Welcome to Real-Time Fake News Detector!**
        
        Click **"🔄 Fetch & Analyze News"** in the sidebar to start monitoring news articles.
        
        **Features:**
        - 📰 Fetch news from 80,000+ sources worldwide
        - 🤖 AI-powered credibility analysis (SVM + LSTM + BERT)
        - 🔍 Filter by country, category, or search query
        - 📊 Real-time statistics and insights
        - 🔄 Auto-refresh every 5 minutes
        """)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8rem;'>
        <p>Powered by NewsAPI & AI Models</p>
        <p>SVM • LSTM • DistilBERT</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
