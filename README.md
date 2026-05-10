# RACE Reading Comprehension & Quiz Generation System

A machine learning pipeline for automated reading comprehension and quiz generation built on the RACE dataset using scikit-learn.

## 📋 Overview

This system comprises:
- **Model A (Supervised)**: Weighted ensemble (SVM×0.6 + LR×0.4) for answer verification
- **Model A (Unsupervised)**: K-Means clustering (k=2) for unsupervised answer verification
- **Model B**: Distractor generation and hint extraction using Logistic Regression rankers
- **Streamlit UI**: Interactive web application with terminal CLI aesthetic

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `streamlit` - Web application framework
- `scikit-learn` - Machine learning models
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scipy` - Sparse matrices
- `joblib` - Model serialization
- `nltk` - BLEU/METEOR metrics
- `rouge-score` - ROUGE metrics

### 2. Run the Streamlit Application

```bash
streamlit run app_streamlit.py
```

The app will open in your browser at `http://localhost:8501`

### 3. Generate the Report

```bash
python generate_report.py
```

This will create `final_report.md` with comprehensive evaluation metrics.

### 4. Generate Visualizations

```bash
python generate_visualizations.py
```

This will create 6 publication-quality graphs in the `results/` folder:
- Model A performance metrics
- Model B performance metrics
- Combined metrics comparison
- Metrics heatmap
- Radar chart comparison
- Dataset statistics

## 📁 Project Structure

```
.
├── app_streamlit.py           # Main Streamlit web application
├── app_cli.py                 # Command-line interface (optional)
├── generate_report.py         # Report generation script
├── final_report.md            # Generated comprehensive report
├── STREAMLIT_README.md        # Detailed Streamlit usage guide
├── data/                      # RACE dataset
│   ├── raw/                   # Original CSV files (train/val/test)
│   └── processed/             # Preprocessed features
├── models/                    # Trained models
│   ├── model_a/traditional/   # LR, SVM, RF models + vectorizer
│   └── model_b/traditional/   # Distractor ranker, hint scorer
├── pipeline/                  # Training and inference code
│   ├── preprocessing.py       # Data preprocessing
│   ├── model_a_train.py       # Model A training
│   ├── model_b_train.py       # Model B training
│   ├── evaluate.py            # Evaluation metrics
│   └── inference.py           # Inference functions
├── notebooks/                 # Jupyter notebooks for EDA
└── results/                   # Generated plots and metrics
```

## 🎯 Features

### Model A - Answer Verification

#### Supervised Models
- Binary classification per option (4 binary predictions → select best)
- Features: One-Hot encoding (5000 tokens) + cosine similarity + 5 lexical features
- Ensemble: SVM×0.6 + LR×0.4 (optimized weights)

#### Unsupervised Model (K-Means)
- K-Means clustering with k=2 (discovers latent answer patterns)
- Same feature space as supervised models (5,006 features)
- Post-hoc cluster assignment based on correctness ratio
- Achieves 32.03% MCQ accuracy (only 1.12pp below best supervised model)

### Model B - Distractor & Hint Generation
- **Distractor Generation**: Dynamic n-gram extraction with Logistic Regression ranker
  - Adapts phrase length to match answer length
  - 5 features: cosine similarities, frequency, char match, length ratio
  - Diversity penalty (cosine < 0.8) for variety
- **Hint Extraction**: Graduated disclosure with Logistic Regression scorer
  - 3 hint levels: general → near-explicit
  - Features: question/answer overlap, position, length

### Streamlit UI
- Terminal CLI aesthetic (phosphor-monitor green on black)
- Interactive quiz interface with MCQ options
- Graduated hint system
- Real-time answer verification
- Analytics dashboard with BLEU/ROUGE/METEOR metrics

## 📊 Performance

Evaluated on test set (8,735 questions):

| Model | MCQ Accuracy | BLEU | ROUGE-1 | ROUGE-L | METEOR |
|-------|--------------|------|---------|---------|--------|
| SVM (Supervised) | **33.15%** | 0.302 | 0.466 | 0.459 | 0.404 |
| Ensemble (Supervised) | 32.63% | 0.297 | 0.462 | 0.455 | 0.400 |
| **K-Means (Unsupervised)** | **32.03%** | 0.303 | 0.459 | 0.451 | 0.404 |
| LR (Supervised) | 32.31% | 0.294 | 0.459 | 0.453 | 0.396 |
| RF (Supervised) | 32.65% | 0.289 | 0.465 | 0.458 | 0.396 |
| Model B (Distractor) | — | 0.026 | 0.134 | 0.126 | 0.076 |
| Model B (Hints) | — | 0.203 | 0.292 | 0.280 | 0.260 |

**Key Finding**: K-Means (unsupervised) achieves competitive performance (32.03%) without using labels during training, demonstrating that feature engineering is more critical than supervised learning for this task.

**Note**: Low BLEU for distractors is expected — short n-grams rarely match exact reference wording.

## 🔧 Training Models

### Preprocess Data
```bash
python -c "from pipeline.preprocessing import preprocess_and_build; preprocess_and_build()"
```

### Train Model A (Supervised)
```bash
python pipeline/model_a_train.py
```

### Train K-Means (Unsupervised)
```bash
python evaluate_kmeans.py
```

### Train Model B
```bash
python pipeline/model_b_train.py
```

### Compare All Models
```bash
python compare_all_models_with_kmeans.py
```

## 📖 Usage Examples

### CLI Mode
```bash
python app_cli.py
```

### Streamlit Mode
```bash
streamlit run app_streamlit.py
```

Features:
1. **Load Random Article**: Fetch a random RACE article
2. **Generate Question**: Auto-generate MCQ from article
3. **Submit Answer**: Verify your answer with Model A
4. **Get Hints**: Request graduated hints (3 levels)
5. **View Metrics**: See BLEU/ROUGE/METEOR scores

## 🎨 UI Theme

Terminal CLI aesthetic with:
- **Colors**: Pure black (#0a0a0a) + phosphor green (#33ff00)
- **Font**: JetBrains Mono (monospaced)
- **Effects**: CRT scanlines, text glow, ASCII art
- **Elements**: Bracket-wrapped buttons `[ ]`, status prefixes `[ OK ]`/`[ ERR ]`

## 📚 Dataset

**RACE** (ReAding Comprehension from Examinations)
- ~87,866 questions from Chinese middle/high school English exams
- Train: 70,258 | Val: 8,859 | Test: 8,735
- Average article length: ~275 words
- Answer balance: ~25% each (A/B/C/D)

## 🔬 Technical Details

### Model A Architecture
- **Vectorization**: Binary CountVectorizer (max_features=5000)
- **Features**: One-Hot + cosine similarity + 5 lexical features (~5,007 dimensions)
- **Models**: 
  - Logistic Regression (saga solver, C=1, max_iter=1000)
  - SVM (LinearSVC + CalibratedClassifierCV, max_iter=2000)
  - Random Forest (200 trees, lexical features only)
- **Ensemble**: Weighted soft-vote (SVM×0.6 + LR×0.4)

### Model B Architecture
- **Vectorization**: Binary CountVectorizer (max_features=3000)
- **Distractor Ranker**: Logistic Regression (max_iter=500)
  - 5 features per candidate
  - Dynamic n-gram sizing (1-8 words based on answer length)
  - Top-3 selection with diversity penalty
- **Hint Scorer**: Logistic Regression (max_iter=500)
  - 4 features: question/answer overlap, position, length

## 📄 Documentation

- `final_report.md` - Comprehensive academic report with evaluation
- `STREAMLIT_README.md` - Detailed Streamlit usage guide
- `DISTRACTOR_GENERATION_EXPLAINED.md` - Technical explanation of distractor pipeline
- `HINT_GENERATION_EXPLAINED.md` - Technical explanation of hint extraction

## 🛠️ Requirements

- Python 3.8+
- scikit-learn 1.0+
- streamlit 1.20+
- pandas, numpy, scipy
- nltk, rouge-score

## 📝 License

This project is for educational purposes as part of AI Lab coursework.

## 🙏 Acknowledgments

- RACE dataset: Lai et al. (2017)
- Reference implementations: Kumar et al. (2015), Liang et al. (2018)

---

**Course**: AI Lab (Semester 5)  
**Framework**: scikit-learn (CPU-only, no deep learning)  
**Dataset**: RACE — ReAding Comprehension from Examinations
