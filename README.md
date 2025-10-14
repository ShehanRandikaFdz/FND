# Flask Fake News Detector

A comprehensive fake news detection system built with Flask, featuring three machine learning models (SVM, LSTM, BERT) and real-time NewsAPI verification.

## Features

- **Multi-Model Ensemble**: Combines SVM, LSTM, and BERT models for robust predictions
- **Majority Voting**: Uses ensemble majority voting for final decisions
- **NewsAPI Integration**: Verifies news against real-time online sources
- **Modern UI**: Clean, responsive design with Tailwind CSS and dark mode
- **Session History**: Tracks analysis history with persistent storage
- **Real-time Analysis**: Instant credibility assessment with confidence scores

## Architecture

```
Flask Backend (app.py)
├── ML Models: SVM, LSTM, BERT (ensemble majority voting)
├── NewsAPI: Real-time verification
├── Routes: /, /analyze, /fetch-news, /history
└── Session-based history
```

## Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd fake-news-detector
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   Create a `.env` file in the root directory:
   ```env
   SECRET_KEY=your-secret-key-here
   NEWSAPI_KEY=your-newsapi-key-here
   FLASK_DEBUG=True
   FLASK_HOST=0.0.0.0
   FLASK_PORT=5000
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the application**
   Open your browser and navigate to: `http://localhost:5000`

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SECRET_KEY` | Flask secret key for sessions | `dev-secret-key-change-in-production` |
| `NEWSAPI_KEY` | NewsAPI key for real-time verification | `""` (empty) |
| `FLASK_DEBUG` | Enable Flask debug mode | `True` |
| `FLASK_HOST` | Host to run Flask on | `0.0.0.0` |
| `FLASK_PORT` | Port to run Flask on | `5000` |

### NewsAPI Setup

1. Visit [NewsAPI.org](https://newsapi.org/)
2. Sign up for a free account
3. Get your API key
4. Add it to your `.env` file:
   ```env
   NEWSAPI_KEY=your-api-key-here
   ```

## File Structure

```
├── app.py                     # Main Flask application
├── config.py                  # Configuration settings
├── requirements.txt           # Python dependencies
├── models/                    # Trained ML models
│   ├── new_svm_model.pkl
│   ├── new_svm_vectorizer.pkl
│   ├── lstm_fake_news_model.h5
│   ├── lstm_tokenizer.pkl
│   └── bert_fake_news_model/
├── utils/                     # Utility modules
│   ├── model_loader.py        # ML model loading
│   ├── predictor.py           # Ensemble prediction
│   ├── news_verifier.py       # NewsAPI verification
│   └── text_preprocessor.py   # Text preprocessing
├── credibility_analyzer/      # Advanced analysis
│   ├── credibility_analyzer.py
│   ├── text_preprocessor.py
│   ├── confidence_calibrator.py
│   └── feature_extractor.py
├── verdict_agent/             # Multi-agent decision system
│   └── verdict_agent.py
├── news_apis/                 # NewsAPI integration
│   └── newsapi_client.py
├── templates/                 # HTML templates
│   └── index.html
└── static/                    # Static assets
    └── js/
        └── main.js
```

## API Endpoints

### POST `/analyze`
Analyze text for fake news using ML models and NewsAPI verification.

**Request:**
```json
{
  "text": "Your news text here..."
}
```

**Response:**
```json
{
  "prediction": "FAKE",
  "confidence": 87.5,
  "news_api_results": {
    "found": true,
    "articles": [...],
    "best_match": {...}
  },
  "individual_results": {
    "svm": {...},
    "lstm": {...},
    "bert": {...}
  },
  "explanation": "Analysis indicates...",
  "timestamp": "2024-10-13T22:30:00"
}
```

### POST `/fetch-news`
Fetch latest news articles from NewsAPI and analyze their credibility.

**Request:**
```json
{
  "country": "us",
  "category": "general",
  "page_size": 10
}
```

### GET `/history`
Retrieve analysis history from session storage.

### POST `/clear-history`
Clear analysis history from session storage.

## Machine Learning Models

### Support Vector Machine (SVM)
- **Accuracy**: 99.5%
- **Features**: TF-IDF vectorization
- **Strengths**: High accuracy on structured text features

### Long Short-Term Memory (LSTM)
- **Accuracy**: 87.0%
- **Features**: Sequential text processing
- **Strengths**: Captures temporal patterns in text

### DistilBERT (Hybrid)
- **Accuracy**: 89.0%
- **Features**: Pre-trained DistilBERT + Logistic Regression
- **Strengths**: Excellent context understanding with efficient hybrid approach

## Usage Examples

### Basic Text Analysis
```javascript
// Analyze text
const response = await fetch('/analyze', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ text: 'Your news text here...' })
});
const result = await response.json();
```

### Fetch Latest News
```javascript
// Fetch and analyze latest news
const response = await fetch('/fetch-news', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ country: 'us', category: 'general' })
});
const articles = await response.json();
```

## Deployment

### Local Development
```bash
python app.py
```

### Production Deployment

1. **Set production environment variables**
   ```env
   FLASK_DEBUG=False
   SECRET_KEY=your-production-secret-key
   NEWSAPI_KEY=your-newsapi-key
   ```

2. **Use a production WSGI server**
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

## Troubleshooting

### Common Issues

1. **Models not loading**
   - Ensure model files are in the `models/` directory
   - Check file permissions and paths

2. **NewsAPI errors**
   - Verify your API key is correct
   - Check your internet connection
   - Ensure you haven't exceeded rate limits

3. **Memory issues with BERT**
   - The system automatically uses memory-efficient settings
   - BERT model uses half-precision when possible

### Performance Tips

- The system loads models once at startup for better performance
- Use the session history feature to avoid re-analyzing the same text
- NewsAPI verification is optional and can be disabled if not needed

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NewsAPI.org for real-time news data
- Hugging Face for pre-trained BERT models
- The open-source ML community for various libraries and tools

---

Built with ❤️ using Flask, Tailwind CSS, TensorFlow, and modern web technologies.