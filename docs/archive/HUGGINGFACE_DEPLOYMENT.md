# 🚀 Hugging Face Spaces Deployment Guide

## Overview
This guide addresses compatibility issues when deploying the Fake News Detection system to Hugging Face Spaces.

## Common Issues & Solutions

### 1. NumPy Version Compatibility (`numpy._core` error)

**Problem:**
```
Error loading SVM model: No module named 'numpy._core'
```

**Solution:**
- Fixed in `requirements.txt` by pinning numpy to version 1.23.5
- Added compatibility layer in `utils/compatibility.py` that creates `numpy._core` alias
- Using `safe_load_pickle()` function for all pickle files

**Files Modified:**
- `requirements.txt`: Changed `numpy>=1.22,<=1.24.3` to `numpy==1.23.5`
- `utils/compatibility.py`: New file with compatibility utilities
- `credibility_analyzer/credibility_analyzer.py`: Uses safe loading functions

### 2. TensorFlow/Keras Model Loading (`batch_shape` error)

**Problem:**
```
Error when deserializing class 'InputLayer': Unrecognized keyword arguments: ['batch_shape']
```

**Solution:**
- Load LSTM models with `compile=False` parameter
- Manually compile model after loading with custom configuration
- Use `safe_load_keras_model()` wrapper function

**Files Modified:**
- `utils/compatibility.py`: Added `safe_load_keras_model()` function
- `credibility_analyzer/credibility_analyzer.py`: Uses safe Keras loading

### 3. Transformers/BERT Accelerate Dependency

**Problem:**
```
Error loading BERT model: Using low_cpu_mem_usage=True requires Accelerate
```

**Solution:**
- Set `low_cpu_mem_usage=False` when loading BERT models
- Added `accelerate==0.23.0` to requirements.txt as backup
- Use `safe_load_transformers_model()` wrapper function

**Files Modified:**
- `requirements.txt`: Added `accelerate==0.23.0`
- `utils/compatibility.py`: Added `safe_load_transformers_model()` function
- `utils/model_loader.py`: Updated BERT loading logic

## Updated Requirements

```txt
streamlit==1.28.1
torch==2.0.1
transformers==4.33.2
tensorflow==2.13.1
scikit-learn==1.3.0
pandas==2.0.3
numpy==1.23.5
plotly==5.17.0
protobuf==3.20.3
joblib==1.3.2
accelerate==0.23.0
```

## Deployment Checklist

### Pre-Deployment
- [ ] Update `requirements.txt` with pinned versions
- [ ] Test compatibility layer locally
- [ ] Ensure all model files are uploaded to Hugging Face Space
- [ ] Check file paths are relative (not absolute)

### Model Files Structure
```
models/
├── new_svm_model.pkl
├── new_svm_vectorizer.pkl
├── lstm_fake_news_model.h5
├── lstm_tokenizer.pkl
└── bert_fake_news_model/
    ├── classifier.pkl
    ├── config.json
    ├── tokenizer.json
    └── vocab.txt
```

### Post-Deployment Verification
- [ ] Check startup logs for model loading messages
- [ ] Verify all three models load successfully
- [ ] Test prediction functionality
- [ ] Monitor memory usage

## Graceful Degradation

The system now supports graceful degradation:

1. **All Models Available**: Full ensemble prediction with highest accuracy
2. **Partial Models**: Uses available models with adjusted weights
3. **Single Model**: Falls back to best available model
4. **No Models**: Returns error with helpful message

### Checking Model Status

The system logs model loading status:
```
✅ SVM model loaded         # Success
✅ LSTM model loaded        # Success
✅ BERT model loaded        # Success
⚠️ Model loading failed     # Partial failure (continues)
❌ Model unavailable        # Complete failure (continues)
```

## Troubleshooting

### If models fail to load:

1. **Check file paths**
   ```python
   import os
   print(os.listdir('models'))  # List model files
   ```

2. **Check numpy version**
   ```python
   import numpy as np
   print(np.__version__)  # Should be 1.23.5
   ```

3. **Check dependency versions**
   ```python
   from utils.compatibility import print_dependency_report
   print_dependency_report()
   ```

4. **Test individual model loading**
   ```python
   from utils.compatibility import safe_load_pickle
   model = safe_load_pickle('models/new_svm_model.pkl')
   print("SVM loaded successfully!")
   ```

## Performance Optimization

### Memory Usage
- BERT model uses ~500MB RAM
- LSTM model uses ~200MB RAM
- SVM model uses ~50MB RAM
- Total: ~750MB RAM required

### Speed
- SVM: <100ms per prediction
- LSTM: ~200ms per prediction
- BERT: ~500ms per prediction
- Ensemble: ~800ms per prediction

### Optimization Tips
1. Enable model caching in Streamlit
2. Use `@st.cache_resource` for model loading
3. Batch predictions when possible
4. Consider model quantization for production

## Environment Variables

Optional environment variables for tuning:

```bash
# Disable TensorFlow warnings
TF_CPP_MIN_LOG_LEVEL=2

# PyTorch memory management
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Disable oneDNN for reproducibility
TF_ENABLE_ONEDNN_OPTS=0
```

## Support

If issues persist:
1. Check Hugging Face Spaces logs
2. Review compatibility layer in `utils/compatibility.py`
3. Test locally with same dependency versions
4. Open an issue with full error logs

## Version History

- **v1.0**: Initial deployment
- **v1.1**: Fixed numpy compatibility
- **v1.2**: Fixed TensorFlow/Keras loading
- **v1.3**: Fixed BERT accelerate dependency
- **v1.4**: Added graceful degradation
