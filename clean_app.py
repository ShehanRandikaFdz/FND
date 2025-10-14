"""
Clean Flask Fake News Detector Application
Minimal implementation for testing
"""

import os
import sys
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'test-secret-key'
CORS(app)

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze text for fake news"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Simple mock response
        response = {
            'prediction': 'TRUE',
            'confidence': 75.0,
            'news_api_results': {'found': False, 'articles': [], 'error': None},
            'individual_results': {
                'svm': {'prediction': 'TRUE', 'confidence': 80.0},
                'lstm': {'prediction': 'TRUE', 'confidence': 70.0},
                'bert': {'prediction': 'TRUE', 'confidence': 75.0}
            },
            'timestamp': datetime.now().isoformat(),
            'text': text[:100] + '...' if len(text) > 100 else text,
            'explanation': 'This is a test explanation for the mock response.'
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in analyze: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/fetch-news', methods=['POST'])
def fetch_news():
    """Fetch latest news from NewsAPI"""
    try:
        # Return mock data
        mock_articles = [
            {
                'title': 'Sample News Article 1',
                'description': 'This is a sample news article for testing purposes.',
                'url': 'https://example.com/article1',
                'source': 'Sample News',
                'published_at': '2024-10-13',
                'credibility_score': 0.8,
                'prediction': 'TRUE',
                'confidence': 85
            }
        ]
        return jsonify({'articles': mock_articles})
        
    except Exception as e:
        print(f"Error fetching news: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/history', methods=['GET'])
def get_history():
    """Get analysis history"""
    history = session.get('history', [])
    return jsonify({'history': history})

@app.route('/clear-history', methods=['POST'])
def clear_history():
    """Clear analysis history"""
    session['history'] = []
    session.modified = True
    return jsonify({'message': 'History cleared'})

if __name__ == '__main__':
    print("Starting Clean Flask Fake News Detector...")
    print("Application ready!")
    print("Registered routes:")
    for rule in app.url_map.iter_rules():
        print(f"  {rule}")
    print("Starting server on 0.0.0.0:5000")
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        use_reloader=False,
        threaded=True
    )
