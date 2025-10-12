# 🔍 Fake News Detection System---

title: Fake News Detector

A sophisticated machine learning system that detects fake news using an ensemble of three state-of-the-art models: SVM, LSTM, and BERT. Features include real-time analysis, credibility scoring, and comprehensive explainability.emoji: 🔍

colorFrom: red

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)colorTo: blue

[![Streamlit](https://img.shields.io/badge/streamlit-1.50.0-FF4B4B.svg)](https://streamlit.io)sdk: streamlit

[![TensorFlow](https://img.shields.io/badge/tensorflow-2.20.0-FF6F00.svg)](https://tensorflow.org)sdk_version: 1.28.0

app_file: app.py

---pinned: false

license: mit

## ✨ Features---



### 🤖 Multi-Model Ensemble# 🔍 Fake News Detection System

- **SVM (Support Vector Machine)**: 99.59% accuracy - Traditional ML with TF-IDF

- **LSTM (Long Short-Term Memory)**: 98.90% accuracy - Deep learning for sequential patternsA comprehensive fake news detection application that combines three powerful machine learning models (SVM, LSTM, and BERT) with advanced credibility analysis and verdict generation systems.

- **BERT (Transformer)**: 97.50% accuracy - State-of-the-art contextual understanding

- **Ensemble Accuracy**: 98.66% through intelligent voting## 🚀 Live Demo



### 🎯 Advanced Analysis[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-huggingface-space-url.hf.space)

- **9 Credibility Analyzers**:

  - Bias Detection## ✨ Features

  - Clickbait Detection

  - Emotional Manipulation Analysis- **Multi-Model Analysis**: Combines SVM (99.5% accuracy), LSTM (98.9% accuracy), and DistilBERT (97.5% accuracy)

  - Linguistic Inconsistencies- **Real-time Detection**: Instant analysis with detailed explanations

  - Named Entity Recognition (NER)- **Credibility Assessment**: Advanced risk factor identification

  - Propaganda Detection- **Model Comparison**: Side-by-side performance metrics

  - Sentiment Analysis- **Statistical Insights**: Session analytics and confidence tracking

  - Source Credibility- **Responsive Design**: Works on desktop and mobile devices

  - Tone Analysis- **🆕 Graceful Degradation**: Works even if some models fail to load

- **🆕 Compatibility Layer**: Handles version conflicts automatically

### ⚡ Performance

- **Instant Startup**: <1 second app load time## 🎯 How It Works

- **Fast Predictions**: <0.5 seconds after initial model loading

- **Smart Caching**: Models load once, reused for all predictions### Model Architecture

- **Progress Feedback**: Real-time loading indicators

1. **Support Vector Machine (SVM)**

---   - Uses TF-IDF vectorization

   - Excellent for structured text features

## 🚀 Quick Start   - Highest accuracy model (99.59%)



### Installation2. **Long Short-Term Memory (LSTM)**

   - Deep learning for sequential patterns

1. **Clone the repository**   - Captures temporal context

```powershell   - High accuracy (98.90%)

git clone https://github.com/ShehanRandikaFdz/FND.git

cd FND3. **DistilBERT**

```   - Transformer-based architecture

   - Bidirectional attention mechanism

2. **Install dependencies**   - State-of-the-art text understanding (97.50%)

```powershell

pip install -r requirements.txt### Analysis Pipeline

```

1. **Text Preprocessing**: Clean and normalize input text

3. **Run the application**2. **Individual Predictions**: Each model provides its assessment

```powershell3. **Ensemble Combination**: Weighted voting based on model performance

python -m streamlit run app.py4. **Credibility Analysis**: Risk factor identification and uncertainty quantification

```5. **Final Verdict**: Comprehensive analysis with explanations



4. **Open your browser** at `http://localhost:8501`## 🛠️ Installation



---### Local Development



## 📖 Usage1. Clone the repository:

```bash

### Making a Predictiongit clone https://github.com/yourusername/fake-news-detection.git

cd fake-news-detection

1. **Enter Text** - Type or paste a news article```

2. **Click "Analyze"** - Models load on first use (~5 seconds)

3. **View Results** - Get verdict, confidence, and detailed analysis2. Install dependencies:

```bash

### Examplepip install -r requirements.txt

```

```

Input: "Breaking news: Scientists discover new solar technology with 95% efficiency"3. Download model files and place them in the `models/` directory:

   - `models/new_svm_model.pkl`

Output:   - `models/new_svm_vectorizer.pkl`

- Verdict: TRUE   - `models/lstm_fake_news_model.h5`

- Confidence: 87.3%   - `models/lstm_tokenizer.pkl`

- Agreement: 3/3 models   - `models/bert_fake_news_model/`

```

4. **🆕 Verify environment** (recommended before first run):

---```bash

python verify_environment.py

## 🏗️ Project Structure```



```5. Run the application:

FND/```bash

├── app.py                      # Main applicationstreamlit run app.py

├── models/                     # Pre-trained models```

├── credibility_analyzer/       # Analysis modules (9 analyzers)

├── verdict_agent/              # Decision system### Hugging Face Spaces

├── utils/                      # Utilities & compatibility

├── tests/                      # Testing & benchmarksThe application is deployed on Hugging Face Spaces. For deployment instructions, see [`HUGGINGFACE_DEPLOYMENT.md`](HUGGINGFACE_DEPLOYMENT.md).

└── docs/                       # Documentation

```**🆕 Deployment Checklist:**

- ✅ Compatibility fixes applied

---- ✅ Environment verified

- ✅ All models tested

## 🧪 Testing- ✅ Graceful degradation enabled



```powershellSee [`DEPLOYMENT_QUICK_REFERENCE.md`](DEPLOYMENT_QUICK_REFERENCE.md) for quick deployment guide.

# Test functionality

python tests\test_app_functionality.py## 📊 Performance Metrics



# Verify environment| Model | Accuracy | Precision | Recall | F1-Score |

python tests\verify_environment.py|-------|----------|-----------|--------|----------|

| SVM   | 99.59%   | 99.4%     | 99.6%  | 99.5%    |

# Benchmark performance| LSTM  | 98.90%   | 98.5%     | 99.0%  | 98.7%    |

python tests\benchmark_performance.py| BERT  | 97.50%   | 97.0%     | 98.0%  | 97.5%    |

```

## 🔧 Technical Details

---

### Memory Optimization

## 📊 Performance Metrics- BERT model uses half-precision (float16) for reduced memory usage

- Lazy loading of models with Streamlit caching

| Metric | Value |- Optimized tokenization with reduced sequence length

|--------|-------|- **🆕 Compatibility layer** handles version conflicts

| **App Startup** | <1 second ⚡ |- **🆕 Safe model loading** with error recovery

| **First Prediction** | ~5 seconds |

| **Next Predictions** | <0.5 seconds |### Dependencies

| **Ensemble Accuracy** | 98.66% |- **Streamlit**: Web application framework

| **Memory Usage** | ~750MB |- **TensorFlow**: LSTM model implementation

- **PyTorch**: BERT model implementation

---- **scikit-learn**: SVM model and preprocessing

- **Transformers**: Hugging Face transformers library

## 🔧 Troubleshooting- **Plotly**: Interactive visualizations

- **🆕 Accelerate**: BERT optimization (optional)

### Port Already in Use

```powershell### 🆕 Compatibility Features

python -m streamlit run app.py --server.port 8502- Automatic numpy version compatibility

```- TensorFlow/Keras backward compatibility

- BERT loading without strict accelerate requirement

### Models Not Loading- Graceful degradation when models unavailable

```powershell

python tests\verify_environment.py## 📁 Project Structure

```

```

### Slow First Predictionfake-news-detection/

Normal behavior - models load on first use. Subsequent predictions are instant.├── app.py                           # Main Streamlit application

├── requirements.txt                 # Python dependencies

---├── README.md                        # Project documentation

├── 🆕 verify_environment.py         # Environment verification script

## 📚 Documentation├── 🆕 test_compatibility.py         # Compatibility test suite

├── 🆕 HUGGINGFACE_DEPLOYMENT.md     # Deployment guide

- **[DEPLOYMENT.md](docs/DEPLOYMENT.md)** - Deployment guide├── 🆕 DEPLOYMENT_QUICK_REFERENCE.md # Quick reference card

- **[COMPATIBILITY_FIXES.md](docs/COMPATIBILITY_FIXES.md)** - Technical notes├── 🆕 COMPATIBILITY_FIXES_SUMMARY.md # Technical fixes summary

- **[Archive](docs/archive/)** - Historical docs├── 🆕 CHANGES_REPORT.md             # Complete changes report

├── models/                          # Model files (use Git LFS)

---│   ├── new_svm_model.pkl

│   ├── new_svm_vectorizer.pkl

## 🤝 Contributing│   ├── lstm_fake_news_model.h5

│   ├── lstm_tokenizer.pkl

1. Fork the repository│   └── bert_fake_news_model/

2. Create feature branch (`git checkout -b feature/NewFeature`)├── utils/

3. Commit changes (`git commit -m 'Add NewFeature'`)│   ├── 🆕 compatibility.py   # Version compatibility layer

4. Push to branch (`git push origin feature/NewFeature`)│   ├── model_loader.py       # Model loading with optimization

5. Open Pull Request│   └── predictor.py          # Unified prediction interface

├── credibility_analyzer/     # Advanced credibility analysis

---│   ├── credibility_analyzer.py

│   ├── advanced_fusion.py

## 📝 License│   ├── bias_detector.py

│   ├── fallback_system.py

MIT License - see [LICENSE](LICENSE) file for details.│   ├── input_validator.py

│   ├── performance_optimizer.py

---│   ├── quality_monitor.py

│   ├── robustness_tester.py

## 🙏 Acknowledgments│   └── ... (9 modules total)

└── verdict_agent/           # Multi-agent decision system

**Technologies**: Streamlit, TensorFlow, PyTorch, Transformers, scikit-learn```



**Purpose**: Built to combat misinformation and promote media literacy.## 🎮 Usage



---1. **Main Analysis**: Enter news text to get instant fake news detection

2. **Model Comparison**: Compare performance of different models

## 📞 Support3. **Statistics**: View session analytics and confidence distributions

4. **About**: Learn about the system and its capabilities

- **Issues**: [GitHub Issues](https://github.com/ShehanRandikaFdz/FND/issues)

- **Discussions**: [GitHub Discussions](https://github.com/ShehanRandikaFdz/FND/discussions)### Input Requirements

- Minimum 10 characters

---- Maximum 5000 characters (10,000 for validator)

- Optional title and source fields

## 🎯 Roadmap- **🆕 Automatic sanitization** of malicious inputs



- [ ] API endpoints### Output Features

- [ ] Batch processing- Overall verdict (FAKE/TRUE) with confidence score

- [ ] Multi-language support- Individual model predictions

- [ ] Browser extension- Risk factors identified

- [ ] GPU acceleration- Credibility assessment

- Detailed explanations

---- **🆕 Uncertainty indicators** when confidence is low

- **🆕 Fallback predictions** if models unavailable

## 🎉 Version History

## 🧪 Testing & Verification

### v1.1.0 (Current) - October 2025

- ⚡ 30x faster startup### Pre-Deployment Testing

- 🎨 Improved UI```bash

- 🧹 Cleaned codebase# Verify environment

- 📚 Better documentationpython verify_environment.py



### v1.0.0 - Initial Release# Run compatibility tests

- 🤖 Three-model ensemblepython test_compatibility.py

- 🔍 Nine analyzers```

- 🎯 98.66% accuracy

### Continuous Testing

---The system includes:

- ✅ Adversarial input testing (`robustness_tester.py`)

**Made with ❤️ for better information**- ✅ Bias detection and auditing (`bias_detector.py`)

- ✅ Performance monitoring (`quality_monitor.py`)

**Status**: ✅ Production Ready | **Accuracy**: 98.66% | **Speed**: <0.5s- ✅ Drift detection capabilities

- ✅ Input validation (`input_validator.py`)

---

## ⚠️ Important Disclaimers

*Last Updated: October 12, 2025*

- **Educational Purpose**: This tool is designed for educational and research purposes
- **Not Definitive**: Results should not be the sole basis for important decisions
- **Human Judgment**: Always combine with human judgment and additional verification
- **Bias Awareness**: Models may have biases based on training data
- **Multiple Sources**: Verify information through multiple reliable sources
- **🆕 Graceful Degradation**: System works with reduced functionality if models fail

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Run `verify_environment.py` before committing
- Ensure `test_compatibility.py` passes
- Update documentation as needed
- Follow existing code structure

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