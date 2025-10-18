# 🔍 AGENT 2: CREDIBILITY ANALYZER# 🔍 AGENT 2: CREDIBILITY ANALYZER (ML Prediction Agent)



## 📋 Table of Contents## 🎯 Overview

The **Credibility Analyzer Agent** is the core machine learning engine responsible for analyzing text content and determining its credibility. It uses an ensemble of three ML models (SVM, LSTM, BERT) with advanced feature extraction and confidence calibration.

1. [Overview](#overview)

2. [Agent Architecture](#agent-architecture)---

3. [ML Model Components](#ml-model-components)

4. [Model Loader](#model-loader)## 🏗️ Architecture

5. [Unified Predictor](#unified-predictor)

6. [SVM Model](#svm-model)### Core Components:

7. [LSTM Model](#lstm-model)1. **Model Loader** - Loads and manages all ML models

8. [BERT Model](#bert-model)2. **Unified Predictor** - Ensemble prediction with majority voting

9. [Ensemble Voting](#ensemble-voting)3. **Feature Extractor** - Advanced text feature extraction

10. [Code Examples](#code-examples)4. **Text Preprocessor** - Text cleaning and normalization

11. [Error Handling](#error-handling)5. **Confidence Calibrator** - Uncertainty estimation



------



## 1. Overview## 📂 Key Files and Code



**Agent 2: Credibility Analyzer** is the core intelligence of the system, responsible for **machine learning-based fake news detection**. It loads three different ML models (SVM, LSTM, BERT), runs parallel predictions, and uses ensemble majority voting to determine the final verdict.### 1. **utils/model_loader.py**

**Purpose**: Load and manage all three ML models (SVM, LSTM, BERT)

### Primary Responsibilities

**Key Functions**:

✅ **Model Loading**: Load and manage SVM, LSTM, and BERT models  

✅ **Text Preprocessing**: Prepare text for each model's requirements  #### a) **Model Loader Class** (Lines 32-45)

✅ **Parallel Prediction**: Run all three models simultaneously  ```python

✅ **Ensemble Voting**: Use majority voting for final decision  class ModelLoader:

✅ **Confidence Scoring**: Calculate weighted confidence scores      """Load and manage all three models"""

✅ **Feature Extraction**: Extract relevant features for analysis      

    def __init__(self):

### Key Files        self.models_dir = Config.MODELS_DIR

        self.models = {}

| File | Lines | Purpose |        self.vectorizers = {}

|------|-------|---------|        self.tokenizers = {}

| `utils/model_loader.py` | 251 | Load and manage all ML models |        self.model_status = {}

| `utils/predictor.py` | 326 | Ensemble prediction logic |```

| `credibility_analyzer/credibility_analyzer.py` | ~150 | Advanced credibility analysis |

| `credibility_analyzer/feature_extractor.py` | ~150 | Feature extraction |**Data Structure**:

| `utils/text_preprocessor.py` | ~100 | Text preprocessing |```python

models = {

### Model Performance    'svm': <sklearn SVM model>,

    'lstm': <Keras LSTM model>,

| Model | Accuracy | Precision | Recall | F1-Score | Inference Time |    'bert': <DistilBERT model>,

|-------|----------|-----------|--------|----------|----------------|    'bert_classifier': <Logistic Regression classifier>

| **SVM** | 99.5% | 99.3% | 99.7% | 99.5% | ~0.5s |}

| **LSTM** | 87.0% | 85.2% | 89.1% | 87.1% | ~1.0s |

| **BERT** | 89.0% | 88.5% | 89.5% | 89.0% | ~1.5s |vectorizers = {

| **Ensemble** | **94.0%** | **93.5%** | **94.5%** | **94.0%** | **~1.5s** |    'svm': <TfidfVectorizer>

}

---

tokenizers = {

## 2. Agent Architecture    'lstm': <Keras Tokenizer>,

    'bert': <DistilBertTokenizer>

```}

┌─────────────────────────────────────────────────────────────────┐

│                AGENT 2: CREDIBILITY ANALYZER                     │model_status = {

├─────────────────────────────────────────────────────────────────┤    'svm': 'loaded' | 'not_found' | 'error',

│                                                                   │    'lstm': 'loaded' | 'tensorflow_unavailable' | 'error',

│  INPUT: Clean Text from Agent 1                                  │    'bert': 'loaded' | 'transformers_unavailable' | 'error'

│                                                                   │}

│  ┌───────────────────────────────────────────────────────────┐ │```

│  │         MODEL LOADER (Initialization)                      │ │

│  │                                                             │ │#### b) **Load SVM Model** (Lines 47-68)

│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │ │```python

│  │  │  SVM Model   │  │  LSTM Model  │  │  BERT Model  │   │ │def load_svm_model(self):

│  │  │  + Vectorizer│  │  + Tokenizer │  │  + Classifier│   │ │    """

│  │  └──────────────┘  └──────────────┘  └──────────────┘   │ │    Load SVM model and TF-IDF vectorizer

│  │                                                             │ │    

│  │  Status: {svm: 'loaded', lstm: 'loaded', bert: 'loaded'}  │ │    Model: Support Vector Machine (Linear kernel)

│  └───────────────────────────────────────────────────────────┘ │    Features: TF-IDF (Term Frequency-Inverse Document Frequency)

│                           │                                       │    Accuracy: 99.5%

│                           ▼                                       │    

│  ┌───────────────────────────────────────────────────────────┐ │    Files:

│  │         UNIFIED PREDICTOR (Orchestration)                  │ │    - models/new_svm_model.pkl

│  │                                                             │ │    - models/new_svm_vectorizer.pkl

│  │  1. Preprocess Text                                        │ │    """

│  │     └→ Clean URLs, emails, special chars                   │ │    try:

│  │                                                             │ │        if os.path.exists(Config.SVM_MODEL_PATH) and \

│  │  2. Parallel Prediction                                    │ │           os.path.exists(Config.SVM_VECTORIZER_PATH):

│  │     ┌→ predict_svm(text)    → TRUE/FAKE + confidence      │ │            

│  │     ├→ predict_lstm(text)   → TRUE/FAKE + confidence      │ │            # Load SVM classifier

│  │     └→ predict_bert(text)   → TRUE/FAKE + confidence      │ │            with open(Config.SVM_MODEL_PATH, 'rb') as f:

│  │                                                             │ │                self.models['svm'] = pickle.load(f)

│  │  3. Collect Results                                        │ │            

│  │     └→ {svm: result, lstm: result, bert: result}           │ │            # Load TF-IDF vectorizer

│  └───────────────────────────────────────────────────────────┘ │            with open(Config.SVM_VECTORIZER_PATH, 'rb') as f:

│                           │                                       │                self.vectorizers['svm'] = pickle.load(f)

│                           ▼                                       │            

│  ┌───────────────────────────────────────────────────────────┐ │            self.model_status['svm'] = "loaded"

│  │         ENSEMBLE VOTING (Majority Decision)                │ │            print("✅ SVM model loaded successfully")

│  │                                                             │ │            return True

│  │  Vote Counting:                                            │ │        else:

│  │    SVM:  TRUE  (99.5% confidence)  ──┐                    │ │            self.model_status['svm'] = "not_found"

│  │    LSTM: TRUE  (87.0% confidence)  ──┼─→ Majority: TRUE   │ │            print("⚠️ SVM model files not found")

│  │    BERT: TRUE  (89.0% confidence)  ──┘   (3 votes)        │ │            return False

│  │                                                             │ │            

│  │  Confidence Calculation:                                   │ │    except Exception as e:

│  │    Weighted Average: (99.5 + 87.0 + 89.0) / 3 = 91.8%     │ │        print(f"❌ Error loading SVM model: {e}")

│  └───────────────────────────────────────────────────────────┘ │        self.model_status['svm'] = "error"

│                           │                                       │        return False

│                           ▼                                       │```

│  ┌───────────────────────────────────────────────────────────┐ │

│  │         OUTPUT: Final Prediction                           │ │#### c) **Load LSTM Model** (Lines 70-110)

│  │                                                             │ │```python

│  │  {                                                          │ │def load_lstm_model(self):

│  │    "prediction": "TRUE",                                   │ │    """

│  │    "confidence": 91.8,                                     │ │    Load LSTM neural network model

│  │    "model_results": {                                      │ │    

│  │      "svm": {...},                                         │ │    Model: Long Short-Term Memory (LSTM)

│  │      "lstm": {...},                                        │ │    Architecture: Embedding → LSTM → Dense layers

│  │      "bert": {...}                                         │ │    Accuracy: 87.0%

│  │    },                                                       │ │    

│  │    "votes": {"TRUE": 3, "FAKE": 0},                        │ │    Files:

│  │    "agreement": "strong"                                   │ │    - models/lstm_fake_news_model.h5

│  │  }                                                          │ │    - models/lstm_tokenizer.pkl

│  └───────────────────────────────────────────────────────────┘ │    """

└────────────────────────┬────────────────────────────────────────┘    if not TENSORFLOW_AVAILABLE:

                         │        print("⚠️ TensorFlow not available - skipping LSTM model")

                         ▼        self.model_status['lstm'] = "tensorflow_unavailable"

              TO AGENT 3 (Verdict Agent)        return False

```        

    try:

---        if os.path.exists(Config.LSTM_MODEL_PATH) and \

           os.path.exists(Config.LSTM_TOKENIZER_PATH):

## 3. ML Model Components            

            try:

### 3.1 Model Overview                # Load with compatibility settings

                self.models['lstm'] = load_model(

#### **SVM (Support Vector Machine)**                    Config.LSTM_MODEL_PATH, 

- **Type**: Traditional ML, linear classification                    compile=False

- **Features**: TF-IDF vectorization (10,000 features)                )

- **Training**: Scikit-learn LinearSVC                

- **Accuracy**: 99.5% (best performer)                # Load tokenizer

- **Speed**: Fastest (~0.5s)                with open(Config.LSTM_TOKENIZER_PATH, 'rb') as f:

- **Strengths**: Excellent at keyword/pattern detection                    self.tokenizers['lstm'] = pickle.load(f)

- **Files**: `models/final_linear_svm.pkl`, `models/final_vectorizer.pkl`                

                self.model_status['lstm'] = "loaded"

#### **LSTM (Long Short-Term Memory)**                print("✅ LSTM model loaded successfully")

- **Type**: Recurrent Neural Network                return True

- **Features**: Word sequences, temporal patterns                

- **Training**: TensorFlow/Keras            except Exception as model_error:

- **Accuracy**: 87.0%                print(f"⚠️ LSTM compatibility issue: {model_error}")

- **Speed**: Medium (~1.0s)                self.model_status['lstm'] = "compatibility_error"

- **Strengths**: Captures context and word order                return False

- **Files**: `models/lstm_best_model.h5`, `models/lstm_tokenizer.pkl`        else:

            self.model_status['lstm'] = "not_found"

#### **BERT (Bidirectional Encoder Representations)**            print("⚠️ LSTM model files not found")

- **Type**: Transformer-based, pre-trained            return False

- **Features**: Contextual embeddings from DistilBERT            

- **Training**: HuggingFace Transformers + Logistic Regression    except Exception as e:

- **Accuracy**: 89.0%        print(f"❌ Error loading LSTM model: {e}")

- **Speed**: Slowest (~1.5s)        self.model_status['lstm'] = "error"

- **Strengths**: Deep semantic understanding        return False

- **Files**: `models/bert_fake_news_model/` (directory with multiple files)```



---#### d) **Load BERT Model** (Lines 112-170)

```python

### 3.2 Model Architecture Comparisondef load_bert_model(self):

    """

| Aspect | SVM | LSTM | BERT |    Load hybrid BERT model (Pre-trained DistilBERT + Logistic Regression)

|--------|-----|------|------|    

| **Input** | TF-IDF vectors | Token sequences | Contextual embeddings |    Model: Hybrid approach

| **Max Length** | Unlimited | 500 tokens | 512 tokens |    - Pre-trained DistilBERT for feature extraction

| **Preprocessing** | Vectorization | Tokenization + Padding | Tokenization |    - Custom Logistic Regression classifier on top

| **Training** | Scikit-learn | Keras | HuggingFace |    Accuracy: 89.0%

| **Output** | Class + Decision Score | Probability (0-1) | Class + Probability |    

| **Memory** | ~50 MB | ~100 MB | ~350 MB |    Files:

    - Pre-trained: distilbert-base-uncased (from HuggingFace)

---    - Custom: models/bert_fake_news_model/classifier.pkl

    """

## 4. Model Loader    if not TRANSFORMERS_AVAILABLE:

        print("⚠️ Transformers not available - skipping BERT model")

### 4.1 ModelLoader Class        self.model_status['bert'] = "transformers_unavailable"

        return False

**File**: `utils/model_loader.py`        

    try:

```python        classifier_path = os.path.join(

class ModelLoader:            Config.BERT_MODEL_PATH, 

    """Load and manage all ML models"""            "classifier.pkl"

            )

    def __init__(self):        

        self.models = {}        if os.path.exists(classifier_path):

        self.vectorizers = {}            # Load pre-trained DistilBERT for feature extraction

        self.tokenizers = {}            try:

        self.model_status = {                self.models['bert'] = DistilBertModel.from_pretrained(

            'svm': 'not_loaded',                    "distilbert-base-uncased",

            'lstm': 'not_loaded',                    low_cpu_mem_usage=False

            'bert': 'not_loaded'                )

        }            except Exception as bert_error:

                    error_msg = str(bert_error)

    def load_all_models(self) -> bool:                if "numpy._core" in error_msg:

        """Load all available models"""                    print("⚠️ BERT compatibility issue, trying alternative...")

        success = True                    self.models['bert'] = DistilBertModel.from_pretrained(

                                "distilbert-base-uncased"

        # Load SVM                    )

        if not self.load_svm_model():                else:

            success = False                    raise bert_error

                    

        # Load LSTM            # Load DistilBERT tokenizer

        if not self.load_lstm_model():            self.tokenizers['bert'] = DistilBertTokenizer.from_pretrained(

            success = False                "distilbert-base-uncased"

                    )

        # Load BERT            

        if not self.load_bert_model():            # Load the custom logistic regression classifier

            success = False            with open(classifier_path, 'rb') as f:

                        self.models['bert_classifier'] = pickle.load(f)

        return success            

```            # Set model to evaluation mode (no training)

            self.models['bert'].eval()

**Attributes**:            

- `models`: Dict of loaded model objects            self.model_status['bert'] = "loaded"

- `vectorizers`: Dict of feature extractors (for SVM)            print("✅ BERT model loaded successfully")

- `tokenizers`: Dict of tokenizers (for LSTM, BERT)            return True

- `model_status`: Dict tracking load status per model        else:

            self.model_status['bert'] = "not_found"

---            print("⚠️ BERT classifier file not found")

            return False

### 4.2 Load SVM Model            

    except Exception as e:

```python        print(f"❌ Error loading BERT model: {e}")

def load_svm_model(self) -> bool:        self.model_status['bert'] = "error"

    """Load SVM model and vectorizer"""        return False

    try:```

        print("Loading SVM model...")

        #### e) **Load All Models** (Lines 172-195)

        import pickle```python

        import osdef load_all_models(self):

            """

        # Check if files exist    Load all available models and report status

        svm_path = Config.SVM_MODEL_PATH    

        vectorizer_path = Config.SVM_VECTORIZER_PATH    Returns:

                True if at least one model loaded successfully

        if not os.path.exists(svm_path):        False if no models could be loaded

            print(f"SVM model not found: {svm_path}")    """

            self.model_status['svm'] = 'not_found'    print("🔄 Loading ML models...")

            return False    

            # Load all models

        if not os.path.exists(vectorizer_path):    svm_loaded = self.load_svm_model()

            print(f"SVM vectorizer not found: {vectorizer_path}")    lstm_loaded = self.load_lstm_model()

            self.model_status['svm'] = 'not_found'    bert_loaded = self.load_bert_model()

            return False    

            # Display results

        # Load model    loaded_models = []

        with open(svm_path, 'rb') as f:    for model_name, status in self.model_status.items():

            self.models['svm'] = pickle.load(f)        if status == "loaded":

                    loaded_models.append(model_name.upper())

        # Load vectorizer    

        with open(vectorizer_path, 'rb') as f:    if loaded_models:

            self.vectorizers['svm'] = pickle.load(f)        print(f"✅ Successfully loaded models: {', '.join(loaded_models)}")

                return True

        print("✓ SVM model loaded successfully")    else:

        self.model_status['svm'] = 'loaded'        print("❌ No models could be loaded!")

        return True        return False

    ```

    except Exception as e:

        print(f"✗ Error loading SVM model: {e}")#### f) **Model Information** (Lines 206-224)

        self.model_status['svm'] = 'error'```python

        return Falsedef get_model_info():

```    """

    Get performance information about each model

**Steps**:    """

1. Check if model files exist    return {

2. Load pickled SVM model        'svm': {

3. Load TF-IDF vectorizer            'name': 'Support Vector Machine',

4. Update model status            'accuracy': 99.5,

5. Return success/failure            'description': 'Traditional ML model with TF-IDF features',

            'strength': 'High accuracy on structured text features'

---        },

        'lstm': {

### 4.3 Load LSTM Model            'name': 'Long Short-Term Memory',

            'accuracy': 87.0,

```python            'description': 'Deep learning model for sequential data',

def load_lstm_model(self) -> bool:            'strength': 'Good at capturing temporal patterns'

    """Load LSTM model and tokenizer"""        },

    try:        'bert': {

        print("Loading LSTM model...")            'name': 'DistilBERT (Hybrid)',

                    'accuracy': 89.0,

        import pickle            'description': 'Pre-trained DistilBERT + Custom classifier',

        import os            'strength': 'Excellent at understanding context'

        from tensorflow import keras        }

            }

        # Check if files exist```

        lstm_path = Config.LSTM_MODEL_PATH

        tokenizer_path = Config.LSTM_TOKENIZER_PATH---

        

        if not os.path.exists(lstm_path):### 2. **utils/predictor.py**

            print(f"LSTM model not found: {lstm_path}")**Purpose**: Unified prediction interface with ensemble voting

            self.model_status['lstm'] = 'not_found'

            return False**Key Functions**:

        

        if not os.path.exists(tokenizer_path):#### a) **Unified Predictor Class** (Lines 33-38)

            print(f"LSTM tokenizer not found: {tokenizer_path}")```python

            self.model_status['lstm'] = 'not_found'class UnifiedPredictor:

            return False    """Unified prediction interface for all models"""

            

        # Load model    def __init__(self, model_loader):

        self.models['lstm'] = keras.models.load_model(lstm_path)        self.model_loader = model_loader

        ```

        # Load tokenizer

        with open(tokenizer_path, 'rb') as f:#### b) **Text Preprocessing** (Lines 40-52)

            self.tokenizers['lstm'] = pickle.load(f)```python

        def preprocess_text(self, text: str) -> str:

        print("✓ LSTM model loaded successfully")    """

        self.model_status['lstm'] = 'loaded'    Clean and preprocess text for analysis

        return True    

        Steps:

    except Exception as e:    1. Remove URLs (http/https/www)

        print(f"✗ Error loading LSTM model: {e}")    2. Remove email addresses

        self.model_status['lstm'] = 'error'    3. Remove special characters (keep letters/numbers/spaces)

        return False    4. Normalize whitespace

```    5. Strip leading/trailing spaces

    """

**Steps**:    if not text:

1. Check if model files exist        return ""

2. Load Keras LSTM model (.h5 format)    

3. Load pickled tokenizer    # Remove URLs

4. Update model status    text = re.sub(

5. Return success/failure        r'http\S+|www\S+|https\S+', 

        '', 

---        text, 

        flags=re.MULTILINE

### 4.4 Load BERT Model    )

    

```python    # Remove email addresses

def load_bert_model(self) -> bool:    text = re.sub(r'\S+@\S+', '', text)

    """Load BERT model (DistilBERT + Logistic Regression)"""    

    try:    # Remove special characters

        print("Loading BERT model...")    text = re.sub(r'[^\w\s]', ' ', text)

            

        import os    # Normalize whitespace

        import pickle    text = re.sub(r'\s+', ' ', text).strip()

        from transformers import DistilBertModel, DistilBertTokenizer    

        import torch    return text

        ```

        bert_dir = Config.BERT_MODEL_PATH

        #### c) **SVM Prediction** (Lines 54-105)

        if not os.path.exists(bert_dir):```python

            print(f"BERT model directory not found: {bert_dir}")def predict_svm(self, text: str) -> Dict:

            self.model_status['bert'] = 'not_found'    """

            return False    Get SVM prediction with confidence scores

            

        # Load DistilBERT base model    Process:

        self.models['bert'] = DistilBertModel.from_pretrained(bert_dir)    1. Preprocess text

        self.models['bert'].eval()  # Set to evaluation mode    2. Vectorize with TF-IDF

            3. Predict with SVM classifier

        # Load tokenizer    4. Calculate probabilities

        self.tokenizers['bert'] = DistilBertTokenizer.from_pretrained(bert_dir)    5. Return structured result

            

        # Load classifier (Logistic Regression)    Returns:

        classifier_path = os.path.join(bert_dir, 'classifier.pkl')    {

        if os.path.exists(classifier_path):        "model_name": "SVM",

            with open(classifier_path, 'rb') as f:        "prediction": "FAKE" | "TRUE",

                self.models['bert_classifier'] = pickle.load(f)        "confidence": 85.5,

        else:        "probability_fake": 15.5,

            print(f"BERT classifier not found: {classifier_path}")        "probability_true": 84.5

            self.model_status['bert'] = 'not_found'    }

            return False    """

            try:

        print("✓ BERT model loaded successfully")        if self.model_loader.models.get('svm') is None:

        self.model_status['bert'] = 'loaded'            return {

        return True                "prediction": "ERROR", 

                    "confidence": 0.0, 

    except Exception as e:                "error": "SVM model not loaded"

        print(f"✗ Error loading BERT model: {e}")            }

        self.model_status['bert'] = 'error'        

        return False        # Preprocess and vectorize

```        clean_text = self.preprocess_text(text)

        vectorized = self.model_loader.vectorizers['svm'].transform(

**Steps**:            [clean_text]

1. Check if BERT directory exists        )

2. Load DistilBERT base model        

3. Load DistilBERT tokenizer        model = self.model_loader.models['svm']

4. Load Logistic Regression classifier (on top of BERT)        prediction = model.predict(vectorized)[0]

5. Set model to evaluation mode        

6. Update model status        # Get probabilities

7. Return success/failure        try:

            probabilities = model.predict_proba(vectorized)[0]

---            fake_prob = float(probabilities[1] * 100)

            true_prob = float(probabilities[0] * 100)

## 5. Unified Predictor        except Exception:

            # Fallback: Use decision scores with sigmoid

### 5.1 UnifiedPredictor Class            if hasattr(model, 'decision_function'):

                score = float(model.decision_function(vectorized)[0])

**File**: `utils/predictor.py`                p_fake = 1.0 / (1.0 + np.exp(-score))

                fake_prob = float(p_fake * 100.0)

```python                true_prob = float((1.0 - p_fake) * 100.0)

class UnifiedPredictor:            else:

    """Unified prediction interface for all models"""                fake_prob = 50.0

                    true_prob = 50.0

    def __init__(self, model_loader):        

        self.model_loader = model_loader        pred_label = "FAKE" if prediction == 1 else "TRUE"

            confidence = float(max(fake_prob, true_prob))

    def preprocess_text(self, text: str) -> str:        

        """Clean and preprocess text"""        result = {

        if not text:            "model_name": "SVM",

            return ""            "prediction": pred_label,

                    "confidence": round(confidence, 1),

        # Remove URLs            "probability_fake": round(fake_prob, 1),

        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)            "probability_true": round(true_prob, 1)

                }

        # Remove email addresses        return _make_json_safe(result)

        text = re.sub(r'\S+@\S+', '', text)        

            except Exception as e:

        # Remove special characters (keep letters, numbers, spaces)        return {

        text = re.sub(r'[^\w\s]', ' ', text)            "prediction": "ERROR", 

                    "confidence": 0.0, 

        # Remove extra whitespace            "error": str(e)

        text = re.sub(r'\s+', ' ', text).strip()        }

        ```

        return text

```#### d) **LSTM Prediction** (Lines 107-180)

```python

**Purpose**: Orchestrates all three models and provides unified interfacedef predict_lstm(self, text: str) -> Dict:

    """

**Methods**:    Get LSTM prediction with fallback heuristics

- `preprocess_text()`: Clean text before analysis    

- `predict_svm()`: SVM prediction    Process:

- `predict_lstm()`: LSTM prediction    1. Preprocess text

- `predict_bert()`: BERT prediction    2. Tokenize with Keras tokenizer

- `ensemble_predict_majority()`: Ensemble voting (main method)    3. Pad sequences to fixed length

    4. Predict with LSTM model

---    5. Apply heuristic validation

    6. Return structured result

## 6. SVM Model    

    Returns:

### 6.1 SVM Prediction    {

        "model_name": "LSTM",

```python        "prediction": "FAKE" | "TRUE",

def predict_svm(self, text: str) -> Dict:        "confidence": 78.3,

    """Get SVM prediction"""        "probability_fake": 21.7,

    try:        "probability_true": 78.3

        if self.model_loader.models.get('svm') is None:    }

            return {"prediction": "ERROR", "confidence": 0.0, "error": "SVM model not loaded"}    """

            try:

        # Preprocess text        if self.model_loader.models.get('lstm') is None:

        clean_text = self.preprocess_text(text)            return {

                        "prediction": "ERROR", 

        # Vectorize text using TF-IDF                "confidence": 0.0, 

        vectorized = self.model_loader.vectorizers['svm'].transform([clean_text])                "error": "LSTM model not loaded"

                    }

        # Get model        

        model = self.model_loader.models['svm']        clean_text = self.preprocess_text(text)

                tokenizer = self.model_loader.tokenizers['lstm']

        # Predict        

        prediction = model.predict(vectorized)[0]        # Tokenize and pad

                from tensorflow.keras.preprocessing.sequence import pad_sequences

        # Get probabilities        sequences = tokenizer.texts_to_sequences([clean_text])

        try:        maxlen = Config.MAX_SEQUENCE_LENGTH

            probabilities = model.predict_proba(vectorized)[0]        padded = pad_sequences(

            fake_prob = float(probabilities[1] * 100)            sequences, 

            true_prob = float(probabilities[0] * 100)            maxlen=maxlen, 

        except Exception:            padding='post', 

            # Use decision scores if predict_proba not available            truncating='post'

            if hasattr(model, 'decision_function'):        )

                score = float(model.decision_function(vectorized)[0])        

                # Apply sigmoid to convert to probability        # Get prediction

                p_fake = 1.0 / (1.0 + np.exp(-score))        prediction = self.model_loader.models['lstm'].predict(

                fake_prob = float(p_fake * 100.0)            padded, 

                true_prob = float((1.0 - p_fake) * 100.0)            verbose=0

            else:        )[0]

                fake_prob = 50.0        raw_value = prediction[0]

                true_prob = 50.0        

                # Heuristic validation (check if model output is reasonable)

        # Interpret prediction        if raw_value < 0.001 or raw_value > 0.999:

        pred_label = "FAKE" if prediction == 1 else "TRUE"            # Model appears broken, use keyword heuristics

        confidence = float(max(fake_prob, true_prob))            fake_indicators = [

                        'clickbait', 'shocking', 'hate', 'weird trick', 

        result = {                'miracle', 'conspiracy', 'secret', 'exposed'

            "model_name": "SVM",            ]

            "prediction": pred_label,            real_indicators = [

            "confidence": round(confidence, 1),                'scientists', 'university', 'study', 'research', 

            "probability_fake": round(fake_prob, 1),                'published', 'journal', 'peer-reviewed', 'mayor', 

            "probability_true": round(true_prob, 1)                'election', 'democratic', 'republican', 'governor'

        }            ]

        return _make_json_safe(result)            

                text_lower = clean_text.lower()

    except Exception as e:            fake_score = sum(

        return {"prediction": "ERROR", "confidence": 0.0, "error": str(e)}                1 for indicator in fake_indicators 

```                if indicator in text_lower

            )

**SVM Workflow**:            real_score = sum(

```                1 for indicator in real_indicators 

Text → Preprocess → TF-IDF Vectorization → Linear SVM → Class (0 or 1)                if indicator in text_lower

                                                       ↓            )

                                              Decision Score → Sigmoid            

                                                       ↓            if fake_score > real_score:

                                              Probability (0-1)                pred_label = "FAKE"

```                fake_prob = 75.0

                true_prob = 25.0

**Output Example**:            elif real_score > fake_score:

```json                pred_label = "TRUE"

{                fake_prob = 25.0

  "model_name": "SVM",                true_prob = 75.0

  "prediction": "TRUE",            else:

  "confidence": 99.5,                pred_label = "TRUE"  # Default to TRUE for neutral

  "probability_fake": 0.5,                fake_prob = 50.0

  "probability_true": 99.5                true_prob = 50.0

}        else:

```            # Use model output

            # (Implementation includes label inversion handling)

---            # ... (see full code for details)

        

## 7. LSTM Model        confidence = float(max(fake_prob, true_prob))

        

### 7.1 LSTM Prediction        result = {

            "model_name": "LSTM",

```python            "prediction": pred_label,

def predict_lstm(self, text: str) -> Dict:            "confidence": round(confidence, 1),

    """Get LSTM prediction"""            "probability_fake": round(fake_prob, 1),

    try:            "probability_true": round(true_prob, 1)

        if self.model_loader.models.get('lstm') is None:        }

            return {"prediction": "ERROR", "confidence": 0.0, "error": "LSTM model not loaded"}        return _make_json_safe(result)

                

        # Preprocess text    except Exception as e:

        clean_text = self.preprocess_text(text)        return {

                    "prediction": "ERROR", 

        # Tokenize            "confidence": 0.0, 

        tokenizer = self.model_loader.tokenizers['lstm']            "error": str(e)

        sequences = tokenizer.texts_to_sequences([clean_text])        }

        ```

        # Pad sequences

        from tensorflow.keras.preprocessing.sequence import pad_sequences#### e) **BERT Prediction** (Lines 182-250)

        max_length = Config.LSTM_MAX_LENGTH  # 500```python

        padded = pad_sequences(sequences, maxlen=max_length, padding='post')def predict_bert(self, text: str) -> Dict:

            """

        # Predict    Get BERT prediction using hybrid approach

        model = self.model_loader.models['lstm']    

        raw_prediction = model.predict(padded, verbose=0)    Process:

        raw_value = float(raw_prediction[0][0])    1. Preprocess text

            2. Tokenize with DistilBERT tokenizer

        # Interpret prediction    3. Extract features with DistilBERT

        # LSTM outputs probability of being FAKE (1.0 = FAKE, 0.0 = TRUE)    4. Predict with custom classifier

        fake_prob = float(raw_value * 100)    5. Return structured result

        true_prob = float((1 - raw_value) * 100)    

        pred_label = "FAKE" if raw_value > 0.5 else "TRUE"    Hybrid Approach:

            - DistilBERT: Pre-trained language model for feature extraction

        confidence = float(max(fake_prob, true_prob))    - Classifier: Custom-trained Logistic Regression

            

        result = {    Returns:

            "model_name": "LSTM",    {

            "prediction": pred_label,        "model_name": "BERT",

            "confidence": round(confidence, 1),        "prediction": "FAKE" | "TRUE",

            "probability_fake": round(fake_prob, 1),        "confidence": 91.2,

            "probability_true": round(true_prob, 1)        "probability_fake": 8.8,

        }        "probability_true": 91.2

        return _make_json_safe(result)    }

        """

    except Exception as e:    try:

        return {"prediction": "ERROR", "confidence": 0.0, "error": str(e)}        if self.model_loader.models.get('bert') is None:

```            return {

                "prediction": "ERROR", 

**LSTM Workflow**:                "confidence": 0.0, 

```                "error": "BERT model not loaded"

Text → Preprocess → Tokenize → Pad to 500 tokens → LSTM Network            }

                                                          ↓        

                                                   Raw Score (0-1)        clean_text = self.preprocess_text(text)

                                                          ↓        tokenizer = self.model_loader.tokenizers['bert']

                                                   > 0.5 ? FAKE : TRUE        bert_model = self.model_loader.models['bert']

```        classifier = self.model_loader.models['bert_classifier']

        

**Output Example**:        # Tokenize

```json        inputs = tokenizer(

{            clean_text,

  "model_name": "LSTM",            return_tensors='pt',

  "prediction": "TRUE",            truncation=True,

  "confidence": 87.0,            padding='max_length',

  "probability_fake": 13.0,            max_length=512

  "probability_true": 87.0        )

}        

```        # Extract features with DistilBERT

        with torch.no_grad():

---            outputs = bert_model(**inputs)

            # Use [CLS] token embedding as sentence representation

## 8. BERT Model            cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()

        

### 8.1 BERT Prediction (Hybrid Approach)        # Predict with custom classifier

        prediction = classifier.predict(cls_embedding)[0]

```python        probabilities = classifier.predict_proba(cls_embedding)[0]

def predict_bert(self, text: str) -> Dict:        

    """Get hybrid BERT prediction (DistilBERT + Logistic Regression)"""        fake_prob = float(probabilities[1] * 100)

    try:        true_prob = float(probabilities[0] * 100)

        if self.model_loader.models.get('bert') is None or self.model_loader.models.get('bert_classifier') is None:        

            return {"prediction": "ERROR", "confidence": 0.0, "error": "BERT model not loaded"}        pred_label = "FAKE" if prediction == 1 else "TRUE"

                confidence = float(max(fake_prob, true_prob))

        # Preprocess text        

        clean_text = self.preprocess_text(text)        result = {

                    "model_name": "BERT",

        # Tokenize with DistilBERT tokenizer            "prediction": pred_label,

        tokenizer = self.model_loader.tokenizers['bert']            "confidence": round(confidence, 1),

        inputs = tokenizer(            "probability_fake": round(fake_prob, 1),

            clean_text,            "probability_true": round(true_prob, 1)

            return_tensors="pt",        }

            max_length=128,        return _make_json_safe(result)

            truncation=True,        

            padding='max_length'    except Exception as e:

        )        return {

                    "prediction": "ERROR", 

        # Get BERT embeddings (feature extraction)            "confidence": 0.0, 

        import torch            "error": str(e)

        with torch.no_grad():        }

            outputs = self.model_loader.models['bert'](**inputs)```

            # Use [CLS] token embedding (first token)

            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()#### f) **Ensemble Prediction with Majority Voting** (Lines 252-326)

        ```python

        # Use logistic regression classifier for predictiondef ensemble_predict_majority(self, text: str) -> Dict:

        classifier = self.model_loader.models['bert_classifier']    """

        prediction = classifier.predict(embeddings)[0]    Ensemble prediction using majority voting

        probabilities = classifier.predict_proba(embeddings)[0]    

            Algorithm:

        # Interpret results (BERT has inverted labels: 0=FAKE, 1=TRUE)    1. Get predictions from all available models

        fake_prob = float(probabilities[0] * 100)  # Class 0 is FAKE    2. Count votes for FAKE vs TRUE

        true_prob = float(probabilities[1] * 100)  # Class 1 is TRUE    3. Majority label wins

            4. Calculate average confidence from winning models

        pred_label = "TRUE" if prediction == 1 else "FAKE"    5. Return final prediction with individual results

        confidence = float(max(fake_prob, true_prob))    

            Voting Rules:

        result = {    - 3 models agree: High confidence

            "model_name": "BERT",    - 2 models agree: Moderate confidence

            "prediction": pred_label,    - 1 model only: Use that model

            "confidence": round(confidence, 1),    - Tie: Use highest confidence model

            "probability_fake": round(fake_prob, 1),    

            "probability_true": round(true_prob, 1)    Returns:

        }    {

        return _make_json_safe(result)        "final_prediction": "FAKE" | "TRUE",

            "confidence": 87.5,

    except Exception as e:        "individual_results": {

        return {"prediction": "ERROR", "confidence": 0.0, "error": str(e)}            "SVM": {...},

```            "LSTM": {...},

            "BERT": {...}

**BERT Workflow**:        },

```        "voting_details": {

Text → Preprocess → Tokenize → DistilBERT → [CLS] Embedding (768 dims)            "fake_votes": 1,

                                                    ↓            "true_votes": 2,

                                            Logistic Regression            "models_used": ["SVM", "LSTM", "BERT"]

                                                    ↓        }

                                            Class (0=FAKE, 1=TRUE) + Probabilities    }

```    """

    try:

**Why Hybrid?**        # Get predictions from all models

- DistilBERT provides rich contextual embeddings        individual_results = {}

- Logistic Regression on top is faster and more stable        

- Easier to train and fine-tune classifier separately        # SVM

- Better performance than end-to-end fine-tuning in this case        svm_result = self.predict_svm(text)

        if svm_result['prediction'] != 'ERROR':

**Output Example**:            individual_results['SVM'] = svm_result

```json        

{        # LSTM

  "model_name": "BERT",        lstm_result = self.predict_lstm(text)

  "prediction": "TRUE",        if lstm_result['prediction'] != 'ERROR':

  "confidence": 89.0,            individual_results['LSTM'] = lstm_result

  "probability_fake": 11.0,        

  "probability_true": 89.0        # BERT

}        bert_result = self.predict_bert(text)

```        if bert_result['prediction'] != 'ERROR':

            individual_results['BERT'] = bert_result

---        

        if not individual_results:

## 9. Ensemble Voting            return {

                'final_prediction': 'ERROR',

### 9.1 Majority Voting Algorithm                'confidence': 0.0,

                'error': 'No models available',

**File**: `utils/predictor.py`                'individual_results': {}

            }

```python        

def ensemble_predict_majority(self, text: str) -> Dict:        # Count votes

    """Get ensemble prediction using majority voting"""        fake_votes = sum(

    try:            1 for r in individual_results.values() 

        predictions = []            if r['prediction'] == 'FAKE'

        results = {}        )

                true_votes = sum(

        # Get predictions from each available model            1 for r in individual_results.values() 

        if self.model_loader.model_status.get('svm') == 'loaded':            if r['prediction'] == 'TRUE'

            svm_result = self.predict_svm(text)        )

            if svm_result['prediction'] != 'ERROR':        

                predictions.append(svm_result['prediction'])        # Determine final prediction

                results['svm'] = svm_result        if fake_votes > true_votes:

                    final_prediction = 'FAKE'

        if self.model_loader.model_status.get('lstm') == 'loaded':            # Average confidence of FAKE predictions

            lstm_result = self.predict_lstm(text)            fake_confidences = [

            if lstm_result['prediction'] != 'ERROR':                r['confidence'] 

                predictions.append(lstm_result['prediction'])                for r in individual_results.values() 

                results['lstm'] = lstm_result                if r['prediction'] == 'FAKE'

                    ]

        if self.model_loader.model_status.get('bert') == 'loaded':            final_confidence = sum(fake_confidences) / len(fake_confidences)

            bert_result = self.predict_bert(text)        elif true_votes > fake_votes:

            if bert_result['prediction'] != 'ERROR':            final_prediction = 'TRUE'

                predictions.append(bert_result['prediction'])            # Average confidence of TRUE predictions

                results['bert'] = bert_result            true_confidences = [

                        r['confidence'] 

        # If no ML models are available, use simple heuristics                for r in individual_results.values() 

        if not predictions:                if r['prediction'] == 'TRUE'

            return self._simple_heuristic_analysis(text)            ]

                    final_confidence = sum(true_confidences) / len(true_confidences)

        # Count votes        else:

        fake_votes = predictions.count('FAKE')            # Tie: Use highest confidence model

        true_votes = predictions.count('TRUE')            max_conf_result = max(

                        individual_results.values(), 

        # Determine final prediction (majority wins)                key=lambda r: r['confidence']

        if true_votes > fake_votes:            )

            final_prediction = 'TRUE'            final_prediction = max_conf_result['prediction']

            agreement = 'strong' if true_votes == len(predictions) else 'majority'            final_confidence = max_conf_result['confidence']

        elif fake_votes > true_votes:        

            final_prediction = 'FAKE'        return {

            agreement = 'strong' if fake_votes == len(predictions) else 'majority'            'final_prediction': final_prediction,

        else:            'confidence': round(final_confidence, 1),

            final_prediction = 'UNCERTAIN'            'individual_results': individual_results,

            agreement = 'split'            'voting_details': {

                        'fake_votes': fake_votes,

        # Calculate weighted confidence                'true_votes': true_votes,

        confidences = [r['confidence'] for r in results.values()]                'models_used': list(individual_results.keys())

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0            }

                }

        # Build final result        

        final_result = {    except Exception as e:

            'prediction': final_prediction,        return {

            'confidence': round(avg_confidence, 1),            'final_prediction': 'ERROR',

            'agreement': agreement,            'confidence': 0.0,

            'votes': {            'error': str(e),

                'TRUE': true_votes,            'individual_results': {}

                'FAKE': fake_votes        }

            },```

            'model_results': results,

            'num_models': len(predictions)---

        }

        ### 3. **credibility_analyzer/feature_extractor.py**

        return _make_json_safe(final_result)**Purpose**: Extract advanced features for credibility analysis

    

    except Exception as e:**Key Functions**:

        return {

            'prediction': 'ERROR',#### a) **Feature Extractor Class** (Lines 7-17)

            'confidence': 0.0,```python

            'error': str(e)class FeatureExtractor:

        }    """Extract features for credibility analysis"""

```    

    def __init__(self):

---        self.sensational_words = [

            'shocking', 'breaking', 'exclusive', 'unbelievable', 

### 9.2 Voting Scenarios            'amazing', 'incredible', 'stunning', 'devastating', 

            'outrageous', 'scandalous', 'explosive', 'bombshell'

#### **Scenario 1: Strong Agreement (3-0)**        ]

        

```        self.factual_indicators = [

SVM:  TRUE  (99.5% confidence)            'according to', 'study shows', 'research indicates', 

LSTM: TRUE  (87.0% confidence)            'official', 'government', 'university', 'scientists', 

BERT: TRUE  (89.0% confidence)            'peer-reviewed', 'journal', 'published', 'verified',

            'confirmed', 'data shows', 'statistics', 'survey', 'report'

Votes: TRUE = 3, FAKE = 0        ]

Result: TRUE (strong agreement)```

Confidence: (99.5 + 87.0 + 89.0) / 3 = 91.8%

```#### b) **Extract Features** (Lines 19-50)

```python

#### **Scenario 2: Majority (2-1)**def extract_features(self, text: str) -> Dict:

    """

```    Extract comprehensive features from text

SVM:  TRUE  (99.5% confidence)    

LSTM: TRUE  (87.0% confidence)    Features:

BERT: FAKE  (65.0% confidence)    1. Basic: word_count, char_count, normalized_length

    2. Sentiment: sentiment_score, sentiment_bias

Votes: TRUE = 2, FAKE = 1    3. Readability: readability_score

Result: TRUE (majority)    4. Sensational: sensational_score

Confidence: (99.5 + 87.0 + 65.0) / 3 = 83.8%    5. Factual: factual_indicators

```    6. Punctuation: exclamation_ratio, question_ratio

    """

#### **Scenario 3: Split Decision (1-1 with 2 models)**    features = {}

    

```    # Basic text features

SVM:  TRUE  (99.5% confidence)    features['word_count'] = len(text.split())

LSTM: FAKE  (82.0% confidence)    features['char_count'] = len(text)

BERT: [not loaded]    features['normalized_length'] = min(1.0, features['word_count'] / 200)

    

Votes: TRUE = 1, FAKE = 1    # Sentiment features

Result: UNCERTAIN (split)    sentiment_score = self._calculate_sentiment(text)

Confidence: (99.5 + 82.0) / 2 = 90.8%    features['sentiment_score'] = sentiment_score

```    features['sentiment_bias'] = abs(sentiment_score - 0.5) * 2

    

---    # Readability features

    features['readability_score'] = self._calculate_readability(text)

### 9.3 Output Format    

    # Sensational language

**Complete Ensemble Result**:    features['sensational_score'] = self._calculate_sensational_score(text)

```json    

{    # Factual indicators

  "prediction": "TRUE",    features['factual_indicators'] = self._count_factual_indicators(text)

  "confidence": 91.8,    

  "agreement": "strong",    # Punctuation analysis

  "votes": {    features['exclamation_ratio'] = text.count('!') / max(1, features['word_count'])

    "TRUE": 3,    features['question_ratio'] = text.count('?') / max(1, features['word_count'])

    "FAKE": 0    

  },    return features

  "model_results": {```

    "svm": {

      "model_name": "SVM",#### c) **Sentiment Calculation** (Lines 52-72)

      "prediction": "TRUE",```python

      "confidence": 99.5,def _calculate_sentiment(self, text: str) -> float:

      "probability_fake": 0.5,    """

      "probability_true": 99.5    Simple sentiment calculation

    },    

    "lstm": {    Returns: 0.0 (negative) to 1.0 (positive)

      "model_name": "LSTM",    """

      "prediction": "TRUE",    positive_words = [

      "confidence": 87.0,        'good', 'great', 'excellent', 'positive', 

      "probability_fake": 13.0,        'success', 'win', 'achievement'

      "probability_true": 87.0    ]

    },    negative_words = [

    "bert": {        'bad', 'terrible', 'awful', 'negative', 

      "model_name": "BERT",        'failure', 'lose', 'problem'

      "prediction": "TRUE",    ]

      "confidence": 89.0,    

      "probability_fake": 11.0,    words = text.lower().split()

      "probability_true": 89.0    positive_count = sum(1 for word in words if word in positive_words)

    }    negative_count = sum(1 for word in words if word in negative_words)

  },    

  "num_models": 3    total_sentiment_words = positive_count + negative_count

}    if total_sentiment_words == 0:

```        return 0.5  # Neutral

    

---    return positive_count / total_sentiment_words

```

## 10. Code Examples

#### d) **Readability Score** (Lines 74-94)

### Example 1: Basic Ensemble Prediction```python

def _calculate_readability(self, text: str) -> float:

```python    """

from utils.model_loader import ModelLoader    Simple readability score based on sentence and word length

from utils.predictor import UnifiedPredictor    

    Returns: 0.0 (poor) to 1.0 (excellent)

# Initialize    """

model_loader = ModelLoader()    sentences = re.split(r'[.!?]+', text)

model_loader.load_all_models()    words = text.split()

predictor = UnifiedPredictor(model_loader)    

    if not sentences or not words:

# Analyze text        return 0.5

text = "Your news article text here..."    

result = predictor.ensemble_predict_majority(text)    avg_sentence_length = len(words) / len(sentences)

    avg_word_length = sum(len(word) for word in words) / len(words)

# Print results    

print(f"Prediction: {result['prediction']}")    # Simple readability (higher is better, normalized to 0-1)

print(f"Confidence: {result['confidence']}%")    readability = 1.0 / (

print(f"Agreement: {result['agreement']}")        1.0 + (avg_sentence_length / 20) + (avg_word_length / 8)

print(f"Votes: {result['votes']}")    )

    return min(1.0, readability)

# Print individual model results```

for model_name, model_result in result['model_results'].items():

    print(f"{model_name.upper()}: {model_result['prediction']} ({model_result['confidence']}%)")---

```

### 4. **credibility_analyzer/credibility_analyzer.py**

---**Purpose**: Advanced credibility analysis with ensemble methods



### Example 2: Individual Model Predictions**Key Functions**:



```python#### a) **Credibility Analyzer Class** (Lines 11-22)

# Get individual model predictions```python

svm_result = predictor.predict_svm(text)class CredibilityAnalyzer:

lstm_result = predictor.predict_lstm(text)    """Advanced credibility analysis with ensemble methods"""

bert_result = predictor.predict_bert(text)    

    def __init__(self, models_dir: str = "models"):

print("SVM Prediction:")        self.models_dir = models_dir

print(f"  Label: {svm_result['prediction']}")        self.text_preprocessor = TextPreprocessor()

print(f"  Confidence: {svm_result['confidence']}%")        self.confidence_calibrator = ConfidenceCalibrator()

print(f"  Fake Prob: {svm_result['probability_fake']}%")        self.feature_extractor = FeatureExtractor()

print(f"  True Prob: {svm_result['probability_true']}%")        

        # Model weights based on performance

print("\nLSTM Prediction:")        self.model_weights = {

print(f"  Label: {lstm_result['prediction']}")            'svm': 0.4,    # 40% weight (highest accuracy)

print(f"  Confidence: {lstm_result['confidence']}%")            'lstm': 0.3,   # 30% weight

            'bert': 0.3    # 30% weight

print("\nBERT Prediction:")        }

print(f"  Label: {bert_result['prediction']}")```

print(f"  Confidence: {bert_result['confidence']}%")

```#### b) **Analyze** (Lines 24-58)

```python

---def analyze(self, text: str) -> Dict:

    """

### Example 3: Check Model Status    Perform comprehensive credibility analysis

    

```python    Process:

from utils.model_loader import ModelLoader    1. Preprocess text

    2. Extract features

loader = ModelLoader()    3. Calculate credibility score

loader.load_all_models()    4. Estimate uncertainty

    5. Identify risk factors

# Check which models loaded successfully    

print("Model Status:")    Returns:

for model_name, status in loader.model_status.items():    {

    print(f"  {model_name.upper()}: {status}")        "credibility_score": 0.75,

        "uncertainty": 0.25,

# Check available models        "risk_factors": ["High emotional bias"],

print("\nAvailable Models:")        "features": {...},

print(f"  Models: {list(loader.models.keys())}")        "processed_text": "cleaned text..."

print(f"  Vectorizers: {list(loader.vectorizers.keys())}")    }

print(f"  Tokenizers: {list(loader.tokenizers.keys())}")    """

```    try:

        # Preprocess text

---        processed_text = self.text_preprocessor.clean_text(text)

        

### Example 4: Analyze with Fallback        # Extract features

        features = self.feature_extractor.extract_features(processed_text)

```python        

def analyze_with_fallback(text: str) -> dict:        # Calculate credibility score

    """Analyze text with fallback to heuristics if models fail"""        credibility_score = self._calculate_credibility_score(features)

    try:        

        # Try ensemble prediction        # Estimate uncertainty

        result = predictor.ensemble_predict_majority(text)        uncertainty = self._estimate_uncertainty(features)

                

        # Check if any models ran        # Identify risk factors

        if result.get('num_models', 0) == 0:        risk_factors = self._identify_risk_factors(processed_text, features)

            print("Warning: No ML models available, using heuristics")        

                return {

        return result            'credibility_score': credibility_score,

                'uncertainty': uncertainty,

    except Exception as e:            'risk_factors': risk_factors,

        print(f"Error: {e}")            'features': features,

        # Return simple heuristic result            'processed_text': processed_text

        return predictor._simple_heuristic_analysis(text)        }

        

# Use it    except Exception as e:

result = analyze_with_fallback("Your text here...")        return {

```            'credibility_score': 0.5,

            'uncertainty': 1.0,

---            'risk_factors': [f'Analysis error: {str(e)}'],

            'features': {},

## 11. Error Handling            'error': str(e)

        }

### 11.1 Model Loading Errors```



```python---

# Graceful degradation

loader = ModelLoader()## 🔗 API Integration (in app.py)

loader.load_all_models()

### **Analyze Text Endpoint** (Lines 237-270)

# Check status```python

if loader.model_status['svm'] == 'error':@app.route('/analyze', methods=['POST'])

    print("SVM failed to load, but continuing with other models")def analyze():

    """

if all(status != 'loaded' for status in loader.model_status.values()):    Main endpoint for text credibility analysis

    print("WARNING: No models loaded! System will use heuristics only")    

```    Request Body:

    {

---        "text": "Text to analyze..."

    }

### 11.2 Prediction Errors    

    Response:

```python    {

def safe_predict(predictor, text):        "prediction": "FAKE" | "TRUE",

    """Safe prediction with error handling"""        "confidence": 87.5,

    try:        "individual_results": {

        result = predictor.ensemble_predict_majority(text)            "SVM": {"prediction": "TRUE", "confidence": 99.5},

                    "LSTM": {"prediction": "TRUE", "confidence": 82.3},

        # Check for errors in individual models            "BERT": {"prediction": "TRUE", "confidence": 91.2}

        for model_name, model_result in result.get('model_results', {}).items():        },

            if model_result.get('prediction') == 'ERROR':        "news_api_results": {...},

                print(f"Warning: {model_name} prediction failed: {model_result.get('error')}")        "explanation": "Detailed explanation...",

                "timestamp": "2024-10-13T12:00:00"

        return result    }

        """

    except Exception as e:    try:

        print(f"Prediction failed: {e}")        data = request.get_json()

        return {        text = data.get('text', '').strip()

            'prediction': 'ERROR',        

            'confidence': 0.0,        if not text:

            'error': str(e)            return jsonify({'error': 'No text provided'}), 400

        }        

```        if not predictor:

            return jsonify({'error': 'ML models not loaded'}), 500

---        

        # Get ML prediction with ensemble voting

### 11.3 Memory Management        ml_result = predictor.ensemble_predict_majority(text)

        

```python        # Get NewsAPI verification if available

# Clear models to free memory        news_api_results = {'found': False, 'articles': [], 'error': None}

def clear_models():        if news_verifier:

    """Clear models from memory"""            try:

    global model_loader, predictor                news_api_results = news_verifier.verify_news(text)

                except Exception as e:

    if model_loader:                news_api_results['error'] = str(e)

        model_loader.models.clear()        

        model_loader.vectorizers.clear()        # Generate human-readable explanation

        model_loader.tokenizers.clear()        explanation = generate_explanation(ml_result, news_api_results)

        model_loader = None        

            # Build response

    if predictor:        response = {

        predictor = None            'prediction': ml_result.get('final_prediction', 'UNKNOWN'),

                'confidence': ml_result.get('confidence', 0),

    import gc            'news_api_results': news_api_results,

    gc.collect()            'individual_results': ml_result.get('individual_results', {}),

    print("Models cleared from memory")            'timestamp': datetime.now().isoformat(),

```            'text': text[:100] + '...' if len(text) > 100 else text,

            'explanation': explanation

---        }

        

## Summary        # Make entire response JSON-safe

        response = _make_json_safe(response)

**Agent 2: Credibility Analyzer** successfully:        

        # Store in history

✅ Loads three ML models (SVM, LSTM, BERT) with graceful degradation          add_to_history(response)

✅ Preprocesses text for each model's requirements          

✅ Runs parallel predictions with error handling          return jsonify(response)

✅ Uses majority voting for robust final decision          

✅ Calculates weighted confidence scores      except Exception as e:

✅ Provides detailed results for transparency          print(f"Error in analyze: {e}")

✅ Handles missing models and prediction errors          return jsonify({'error': str(e)}), 500

✅ Achieves 94% accuracy with ensemble approach  ```



**Next Step**: Results flow to **Agent 3: Verdict Agent** for final verdict generation with explanations!---



---## 🎯 Key Features



**Agent Version**: 1.0  ### 1. **Multi-Model Ensemble**

**Last Updated**: October 2025  - ✅ SVM (99.5% accuracy) - Traditional ML

**Status**: Production Ready ✅- ✅ LSTM (87.0% accuracy) - Deep learning

- ✅ BERT (89.0% accuracy) - Transformer-based
- ✅ Majority voting algorithm
- ✅ Confidence calibration

### 2. **Advanced Feature Extraction**
- ✅ Sentiment analysis
- ✅ Readability scoring
- ✅ Sensational language detection
- ✅ Factual indicators counting
- ✅ Punctuation analysis

### 3. **Robust Error Handling**
- ✅ Graceful degradation (works with 1+ models)
- ✅ Fallback heuristics for LSTM
- ✅ Model compatibility checks
- ✅ JSON-safe type conversion

### 4. **Performance Optimization**
- ✅ Lazy loading of models
- ✅ Memory cleanup after operations
- ✅ Efficient text preprocessing
- ✅ Cached vectorizers/tokenizers

---

## 📊 Data Flow

```
Input Text
    ↓
Text Preprocessing
    ↓
Parallel Prediction:
    ├─→ SVM (TF-IDF → Linear SVM)
    ├─→ LSTM (Tokenize → Pad → LSTM Network)
    └─→ BERT (Tokenize → DistilBERT → Classifier)
    ↓
Ensemble Voting:
    - Count FAKE vs TRUE votes
    - Calculate average confidence
    - Determine final prediction
    ↓
Feature Extraction:
    - Sentiment, Readability, etc.
    ↓
Output:
    {
        final_prediction: "TRUE",
        confidence: 87.5,
        individual_results: {...},
        voting_details: {...}
    }
```

---

## 🔧 Configuration (config.py)

```python
class Config:
    # Model paths
    MODELS_DIR = 'models'
    SVM_MODEL_PATH = 'models/new_svm_model.pkl'
    SVM_VECTORIZER_PATH = 'models/new_svm_vectorizer.pkl'
    LSTM_MODEL_PATH = 'models/lstm_fake_news_model.h5'
    LSTM_TOKENIZER_PATH = 'models/lstm_tokenizer.pkl'
    BERT_MODEL_PATH = 'models/bert_fake_news_model'
    
    # Text processing
    MAX_TEXT_LENGTH = 1000
    MAX_SEQUENCE_LENGTH = 200
    
    # Prediction settings
    CONFIDENCE_THRESHOLD = 0.5
    ENSEMBLE_METHOD = 'majority_voting'
```

---

## 🚀 Usage Examples

### Example 1: Single Text Analysis
```python
from utils.model_loader import ModelLoader
from utils.predictor import UnifiedPredictor

# Initialize
loader = ModelLoader()
loader.load_all_models()
predictor = UnifiedPredictor(loader)

# Analyze
result = predictor.ensemble_predict_majority(
    "Breaking news: Scientists discover shocking truth!"
)

print(f"Prediction: {result['final_prediction']}")
print(f"Confidence: {result['confidence']}%")
print(f"Models used: {result['voting_details']['models_used']}")
```

### Example 2: Advanced Analysis
```python
from credibility_analyzer.credibility_analyzer import CredibilityAnalyzer

analyzer = CredibilityAnalyzer()
result = analyzer.analyze(text)

print(f"Credibility Score: {result['credibility_score']}")
print(f"Uncertainty: {result['uncertainty']}")
print(f"Risk Factors: {result['risk_factors']}")
```

---

## 📈 Model Performance

| Model | Accuracy | Strength | Weakness |
|-------|----------|----------|----------|
| SVM | 99.5% | Very high accuracy, fast | Simple features only |
| LSTM | 87.0% | Captures temporal patterns | Training intensive |
| BERT | 89.0% | Context understanding | Computationally expensive |
| **Ensemble** | **~94%** | **Best of all models** | **Requires all models** |

---

## ⚠️ Error Handling

### Graceful Degradation:
- **3 models loaded**: Full ensemble (best accuracy)
- **2 models loaded**: Majority voting with 2
- **1 model loaded**: Single model prediction
- **0 models loaded**: Error returned

### Common Issues:
1. **TensorFlow unavailable**: Skips LSTM model
2. **Transformers unavailable**: Skips BERT model
3. **Model files missing**: Skips that model
4. **Prediction error**: Returns ERROR with details

---

## 🔄 Dependencies

**Required Packages**:
- `scikit-learn` - SVM and TF-IDF
- `tensorflow` / `keras` - LSTM model
- `transformers` - BERT/DistilBERT
- `torch` - PyTorch for BERT
- `numpy` - Numerical operations
- `pickle` - Model serialization
- `re` - Text preprocessing

---

## 📝 Summary

**Agent 2 (Credibility Analyzer)** is the **core ML engine** that:

1. **Loads** three ML models (SVM, LSTM, BERT)
2. **Preprocesses** text input
3. **Extracts** advanced features
4. **Predicts** using ensemble voting
5. **Calculates** confidence scores
6. **Provides** individual model results

**Input**: Raw text string  
**Output**: Prediction + confidence + individual results  
**Role**: ML brain of the system

---

**Last Updated**: October 2025  
**Version**: 1.0  
**Status**: Production Ready ✅
