---
title: Fake News Detector
emoji: 🔍
colorFrom: red
colorTo: blue
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
license: mit
---

# 🔍 Fake News Detection System

A comprehensive fake news detection application that combines three powerful machine learning models (SVM, LSTM, and BERT) with advanced credibility analysis and verdict generation systems.

## 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-huggingface-space-url.hf.space)

## ✨ Features

- **Multi-Model Analysis**: Combines SVM (99.5% accuracy), LSTM (87% accuracy), and DistilBERT (75% accuracy)
- **Real-time Detection**: Instant analysis with detailed explanations
- **Credibility Assessment**: Advanced risk factor identification
- **Model Comparison**: Side-by-side performance metrics
- **Statistical Insights**: Session analytics and confidence tracking
- **Responsive Design**: Works on desktop and mobile devices

## 🎯 How It Works

### Model Architecture

1. **Support Vector Machine (SVM)**
   - Uses TF-IDF vectorization
   - Excellent for structured text features
   - Highest accuracy model

2. **Long Short-Term Memory (LSTM)**
   - Deep learning for sequential patterns
   - Captures temporal context
   - Good for understanding text flow

3. **DistilBERT**
   - Transformer-based architecture
   - Bidirectional attention mechanism
   - State-of-the-art text understanding

### Analysis Pipeline

1. **Text Preprocessing**: Clean and normalize input text
2. **Individual Predictions**: Each model provides its assessment
3. **Ensemble Combination**: Weighted voting based on model performance
4. **Credibility Analysis**: Risk factor identification and uncertainty quantification
5. **Final Verdict**: Comprehensive analysis with explanations

## 🛠️ Installation

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download model files and place them in the `models/` directory:
   - `models/new_svm_model.pkl`
   - `models/new_svm_vectorizer.pkl`
   - `models/lstm_fake_news_model.h5`
   - `models/lstm_tokenizer.pkl`
   - `models/distilbert_fake_news_model/`

4. Run the application:
```bash
streamlit run app.py
```

### Hugging Face Spaces

The application is deployed on Hugging Face Spaces and can be accessed directly through the web interface.

## 📊 Performance Metrics

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| SVM   | 99.5%    | 99.4%     | 99.6%  | 99.5%    |
| LSTM  | 87.0%    | 85.0%     | 89.0%  | 87.0%    |
| BERT  | 75.0%    | 73.0%     | 77.0%  | 75.0%    |

## 🔧 Technical Details

### Memory Optimization
- BERT model uses half-precision (float16) for reduced memory usage
- Lazy loading of models with Streamlit caching
- Optimized tokenization with reduced sequence length

### Dependencies
- **Streamlit**: Web application framework
- **TensorFlow**: LSTM model implementation
- **PyTorch**: BERT model implementation
- **scikit-learn**: SVM model and preprocessing
- **Transformers**: Hugging Face transformers library
- **Plotly**: Interactive visualizations

## 📁 Project Structure

```
fake-news-detection/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── models/                   # Model files (use Git LFS)
│   ├── new_svm_model.pkl
│   ├── new_svm_vectorizer.pkl
│   ├── lstm_fake_news_model.h5
│   ├── lstm_tokenizer.pkl
│   └── distilbert_fake_news_model/
├── utils/
│   ├── model_loader.py       # Model loading with optimization
│   └── predictor.py          # Unified prediction interface
├── credibility_analyzer/     # Advanced credibility analysis
└── verdict_agent/           # Multi-agent decision system
```

## 🎮 Usage

1. **Main Analysis**: Enter news text to get instant fake news detection
2. **Model Comparison**: Compare performance of different models
3. **Statistics**: View session analytics and confidence distributions
4. **About**: Learn about the system and its capabilities

### Input Requirements
- Minimum 10 characters
- Maximum 5000 characters
- Optional title and source fields

### Output Features
- Overall verdict (FAKE/TRUE) with confidence score
- Individual model predictions
- Risk factors identified
- Credibility assessment
- Detailed explanations

## ⚠️ Important Disclaimers

- **Educational Purpose**: This tool is designed for educational and research purposes
- **Not Definitive**: Results should not be the sole basis for important decisions
- **Human Judgment**: Always combine with human judgment and additional verification
- **Bias Awareness**: Models may have biases based on training data
- **Multiple Sources**: Verify information through multiple reliable sources

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face**: For the transformers library and deployment platform
- **Streamlit**: For the amazing web application framework
- **TensorFlow & PyTorch**: For the deep learning frameworks
- **scikit-learn**: For machine learning tools
- **Research Community**: For the datasets and research that made this possible

## 📞 Support

For questions, issues, or suggestions:
- Open an issue on GitHub
- Contact through Hugging Face Spaces
- Email: your-email@example.com

---

**Built with ❤️ using Streamlit, TensorFlow, PyTorch, and Hugging Face**

*Last updated: January 2025*