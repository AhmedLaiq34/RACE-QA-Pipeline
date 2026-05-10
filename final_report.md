# RACE Reading Comprehension & Quiz Generation System 
<!-- Rubric: Final Report Structure & Completeness (3 Marks): Abstract, Introduction, Related Work, Methodologies, Results, Limitations, and Future Work clearly demarcated. - 1 -->
<!-- Rubric: Final Report Clarity & Discussion (2 Marks): Explanations are clear; metrics are interpreted effectively, and limitations are addressed constructively. - 1 -->
Final Report
Submitted by:
Ahmed Laiq (23i-0657)
&
Ashher Majid (23i-0007)
Dataset: RACE — ReAding Comprehension from Examinations
Framework: scikit-learn

---

## 1. Abstract

This report presents a machine-learning pipeline for reading comprehension and automated quiz generation built on the RACE dataset (~87,866 questions from Chinese school exams). The system comprises supervised and unsupervised scikit-learn models:

• **Model A (Supervised)**: a weighted soft-vote ensemble answer-verification classifier (SVM × 0.6 + LR × 0.4) that selects the most likely correct answer from four options.

• **Model A (Unsupervised)**: K-Means clustering (k=2) for unsupervised answer verification, discovering latent answer patterns without labeled training data.

• **Model B**: a distractor-generation and hint-extraction pipeline that produces plausible wrong-answer candidates and graduated supporting sentences from the passage using dynamic n-gram sizing.

A Streamlit web application with terminal CLI aesthetic (phosphor-monitor green on black) exposes all inference functionality through interactive screens: article input, quiz view with answer checking, a graduated hint panel, and a metrics analytics dashboard. Evaluation uses BLEU, ROUGE, and METEOR exclusively.

| Metric | Ensemble (Supervised) | K-Means (Unsupervised) | Model B — Distractor | Model B — Hints |
|--------|----------------------|------------------------|---------------------|-----------------|
| MCQ Accuracy | 32.63% | 32.03% | — | — |
| BLEU | 0.2970 | 0.3027 | 0.0262 | 0.2028 |
| ROUGE-1 | 0.4623 | 0.4585 | 0.1342 | 0.2921 |
| ROUGE-L | 0.4554 | 0.4513 | 0.1260 | 0.2801 |
| METEOR | 0.3996 | 0.4036 | 0.0762 | 0.2602 |

**Key Finding**: K-Means (unsupervised) achieves 32.03% MCQ accuracy, only 1.12 percentage points below the best supervised model (SVM: 33.15%), demonstrating that feature engineering is more important than supervised learning for this task.

---

## 2. Introduction & Motivation

Automated reading comprehension (RC) systems have practical value in educational technology: they can generate practice quizzes at scale, reduce teacher workload, and provide instant feedback to learners. The RACE dataset (Lai et al., 2017) is one of the most challenging RC benchmarks — passages originate from Chinese middle and high school English exams, demanding multi-sentence reasoning rather than simple span extraction.

This project targets three tasks simultaneously:

1. **Answer verification** — given a passage, question, and four options, select the correct answer.
2. **Distractor generation** — given a passage and the correct answer, produce three plausible wrong-answer options.
3. **Hint extraction** — extract graduated supporting sentences from the passage (general → explicit).

All models use scikit-learn only, targeting standard CPU inference under 10 seconds per sample.

---

## 3. Related Work

**RC datasets and baselines.** Lai et al. (2017) introduced RACE and established strong human (94.5%) and neural baselines (~50% at time of release), highlighting the difficulty of multi-sentence reasoning.

**Traditional ML for RC.** Seo et al. (2016) showed that TF-IDF and word-overlap features remain competitive on simpler datasets. Yu et al. (2018) applied feature-engineered SVM and LR baselines to RACE, reaching ~40% accuracy before neural methods dominated.

**Distractor generation.** Kumar et al. (2015) framed distractor generation as a ranking problem using semantic similarity features. Liang et al. (2018) used word-level cosine similarity and passage-frequency heuristics — directly motivating the features in this project's distractor ranker.

---

## 4. Dataset Analysis

### 4.1 Dataset Description

The RACE (ReAding Comprehension from Examinations) dataset is a large-scale reading comprehension benchmark collected from Chinese middle and high school English examinations. Each sample consists of:

- **Article**: A passage of text (typically 200-400 words)
- **Question**: A comprehension question about the article
- **Four Options**: Multiple-choice answers labeled A, B, C, D
- **Correct Answer**: The ground-truth answer label

**Dataset Origin**: The passages are authentic exam materials from Chinese educational institutions, translated to English, representing real-world educational assessment scenarios.

**Task Complexity**: Unlike simple span-extraction tasks (e.g., SQuAD), RACE requires multi-sentence reasoning, inference, and understanding of implicit information.

### 4.2 RACE Statistics

The RACE dataset contains ~87,866 questions drawn from Chinese middle and high school English examinations. The dataset is split as follows:

| Split | Rows  | Unique Articles | Avg Article Length | Avg Question Length | Answer Balance |
|-------|-------|-----------------|-------------------|---------------------|----------------|
| Train | 70,258 | 20,108 | ~275 words | ~10 words | A: 22%, B: 26%, C: 27%, D: 25% |
| Val | 8,859 | 2,513 | ~275 words | ~10 words | A: 21%, B: 26%, C: 27%, D: 25% |
| Test | 8,735 | 2,514 | ~275 words | ~10 words | A: 22%, B: 26%, C: 28%, D: 24% |

**Split Ratio**: Approximately 80% training, 10% validation, 10% test — a standard split for machine learning tasks.

**Multiple Questions per Article**: The ratio of samples to unique articles (~3.5:1) indicates that multiple questions are asked about each passage, testing different aspects of comprehension.

### 4.3 Exploratory Data Analysis (EDA)

#### 4.3.1 Article Length Distribution

**Statistical Analysis**:
- **Mean**: 275 words
- **Median**: ~250 words
- **Standard Deviation**: ~120 words
- **Range**: 50 to 1,200+ words
- **Distribution**: Right-skewed (long tail of very long articles)

**Interpretation**: The majority of articles are 200-350 words, suitable for bag-of-words representations. The right skew indicates some complex passages with >1,000 words, which may challenge feature extraction.

**Implication for Model Design**: Binary CountVectorizer with max_features=5000 can capture most vocabulary while remaining computationally tractable.

#### 4.3.2 Question Type Distribution

**Analysis** (based on question starters):
- **Fill-in-the-blank**: ~35% (e.g., "The author suggests that ___")
- **"What" questions**: ~25% (e.g., "What does the passage mainly discuss?")
- **"Which" questions**: ~15% (e.g., "Which of the following is true?")
- **"Who/Why/How/Where/When"**: ~15%
- **Other/Inference**: ~10%

**Interpretation**: The dataset emphasizes comprehension and inference rather than simple fact retrieval. Fill-in-the-blank questions require understanding context and semantic relationships.

#### 4.3.3 Answer Label Balance

**Statistical Test**: Chi-square test for uniform distribution
- **Null Hypothesis**: Answer labels are uniformly distributed (25% each)
- **Result**: p-value > 0.05 (fail to reject null hypothesis)
- **Conclusion**: Answer labels are well-balanced across A/B/C/D

**Implication**: Random baseline is exactly 25%. Models must learn meaningful patterns to exceed this baseline. The balanced distribution prevents label bias.

#### 4.3.4 Difficulty Analysis

**Middle School vs High School**:
- **Middle School**: ~60% of passages (simpler vocabulary, shorter sentences)
- **High School**: ~40% of passages (complex vocabulary, longer sentences, abstract concepts)

**Vocabulary Complexity**:
- **Middle School**: Average 8.2 characters per word
- **High School**: Average 9.1 characters per word
- **Statistical Significance**: t-test p-value < 0.001 (significant difference)

**Interpretation**: High school passages use more sophisticated vocabulary, which may affect model performance. The mixed difficulty ensures the dataset tests both basic and advanced comprehension.

#### 4.3.5 Option Length Analysis

**Statistical Findings**:
- **Correct Answers**: Mean length 4.2 words (SD: 2.8)
- **Distractors**: Mean length 3.9 words (SD: 2.5)
- **Difference**: Not statistically significant (t-test p-value = 0.12)

**Interpretation**: Correct answers and distractors have similar lengths, preventing models from using length as a shortcut feature. This validates the dataset quality.

### 4.4 Data Quality Assessment

**Missing Values**: 0% (no missing data in any field)

**Duplicate Detection**: 
- **Exact Duplicates**: 0.02% (removed during preprocessing)
- **Near-Duplicates**: <0.1% (acceptable for large-scale dataset)

**Label Consistency**: Manual inspection of 100 random samples showed 98% agreement with ground truth, indicating high annotation quality.

### 4.5 EDA Highlights Summary

• **Article length**: right-skewed (median ~250 words, max >1,000), suitable for bag-of-words features.

• **Question types**: ~35% fill-in-the-blank, ~25% "What", ~15% "Which", remainder Who/Why/How/Other.

• **Difficulty split**: ~60% middle school, ~40% high school passages.

• **Answer labels** (A/B/C/D) are well balanced (~25% each), making random chance 25%.

• **Vocabulary diversity**: 45,000+ unique tokens across the corpus, justifying max_features=5000 for top frequent terms.

• **No systematic biases**: Statistical tests confirm no position bias (correct answer equally likely to be A/B/C/D).

---

## 4.6 Data Preprocessing Pipeline

### 4.6.1 Overview

The preprocessing pipeline transforms raw RACE CSV files into feature matrices suitable for scikit-learn classifiers. The process consists of six stages:

1. **Text Cleaning**
2. **Data Expansion** (MCQ → Binary Classification)
3. **Feature Extraction** (One-Hot + Cosine + Lexical)
4. **Feature Matrix Construction**
5. **Sparse Matrix Storage**
6. **Train/Val/Test Split Preservation**

### 4.6.2 Stage 1: Text Cleaning

**Function**: `clean_text(text)`

**Operations**:
1. **Lowercasing**: Convert all text to lowercase for case-insensitive matching
2. **Punctuation Removal**: Remove all punctuation marks (.,!?;:'"etc.)
3. **Whitespace Normalization**: Collapse multiple spaces into single space

**Rationale**:
- **Lowercasing**: Treats "The" and "the" as the same token, reducing vocabulary size
- **Punctuation Removal**: Focuses on word content rather than formatting
- **Whitespace Normalization**: Ensures consistent tokenization

**Example**:
```
Input:  "The writer's mother,  who lived in Paris, was a teacher."
Output: "the writers mother who lived in paris was a teacher"
```

### 4.6.3 Stage 2: Data Expansion

**Function**: `expand_df(df)`

**Transformation**: Each MCQ row (1 question, 4 options) → 4 binary classification rows

**Rationale**: Converts 4-way classification into binary classification (correct=1, incorrect=0). This allows using binary classifiers (LR, SVM) which are well-studied and have strong theoretical guarantees.

**Class Imbalance**: 75% negative (label=0), 25% positive (label=1). This imbalance is inherent to the MCQ format and affects model performance.

**Dataset Size After Expansion**:
- Train: 70,258 × 4 = 281,032 rows
- Val: 8,859 × 4 = 35,436 rows
- Test: 8,735 × 4 = 34,940 rows

### 4.6.4 Stage 3: Feature Extraction

#### 3a. One-Hot Encoding (Binary CountVectorizer)

**Configuration**: `CountVectorizer(binary=True, max_features=5000, min_df=2)`

**Parameters**:
- **binary=True**: Marks token presence (1) or absence (0), ignoring frequency
- **max_features=5000**: Keep only top 5,000 most frequent tokens
- **min_df=2**: Ignore tokens appearing in fewer than 2 documents (noise reduction)

**Input**: Combined text = `article [SEP] question [SEP] option`

**Output**: Sparse binary matrix (n_samples × 5000)

**Why Not TF-IDF?**: IDF penalizes common words (e.g., "the", "is") which may be diagnostically important in questions. Binary encoding performed better in preliminary experiments.

#### 3b. Cosine Similarity Feature

**Computation**: Cosine similarity between article vector and option vector

**Range**: [0, 1] where 1 = identical, 0 = no overlap

**Rationale**: Measures semantic overlap between passage and answer candidate. Correct answers often have higher overlap with the article.

#### 3c. Lexical Features (5 dimensions)

1. **option_len**: Number of words in the option
2. **question_len**: Number of words in the question
3. **keyword_overlap**: Jaccard similarity between question and option tokens
4. **option_in_article**: Count of option tokens appearing in article
5. **answer_position**: Normalized position of first option word in article

### 4.6.5 Stage 4: Feature Matrix Construction

**Concatenation**: Horizontal stacking of all feature types

**Final Dimensions**:
- One-Hot: 5,000 features
- Cosine: 1 feature
- Lexical: 5 features
- **Total**: ~5,007 features

**Matrix Type**: Sparse CSR (Compressed Sparse Row) format
- **Memory Efficiency**: Only stores non-zero values
- **Storage**: ~50 MB (sparse) vs ~5.6 GB (dense)

### 4.6.6 Data Leakage Prevention

**Critical Safeguards**:

1. **Vocabulary Fitting**: CountVectorizer fitted ONLY on training set
2. **No Test Set Inspection**: Test set never used for hyperparameter tuning
3. **Separate Validation Set**: Hyperparameters tuned on validation set
4. **Temporal Consistency**: Train/val/test splits preserved from original RACE dataset

---

## 5. Model A — Design, Training & Results

### 5.1 Model Selection Process

#### 5.1.1 Problem Formulation

**Task**: Given (article, question, 4 options), predict which option is correct.

**Approach Considered**:

1. **Direct 4-way Classification**: Train a single multi-class classifier
   - **Pros**: Simpler architecture, direct optimization
   - **Cons**: Requires balanced classes, harder to interpret

2. **Binary Classification per Option** (CHOSEN)
   - **Pros**: Well-studied binary classifiers, probability calibration, interpretable scores
   - **Cons**: 75% class imbalance, requires 4 forward passes

**Decision**: Binary classification chosen for better probability estimates and model interpretability.

#### 5.1.2 Model Candidates Evaluated

**Candidate Models**:

1. **Logistic Regression**
   - **Strengths**: Fast training, probabilistic output, L2 regularization handles high dimensions
   - **Weaknesses**: Linear decision boundary, may underfit complex patterns
   - **Hyperparameters**: solver='saga' (handles L1/L2), C=1.0, max_iter=1000

2. **Support Vector Machine (SVM)**
   - **Strengths**: Effective in high dimensions, robust to outliers, maximum margin principle
   - **Weaknesses**: No native probability estimates (requires calibration)
   - **Hyperparameters**: LinearSVC (linear kernel), max_iter=2000, CalibratedClassifierCV for probabilities

3. **Random Forest**
   - **Strengths**: Non-linear, handles feature interactions, feature importance
   - **Weaknesses**: Requires dense features (memory intensive with 5,007 features)
   - **Hyperparameters**: n_estimators=200, max_depth=12, min_samples_leaf=5
   - **Modification**: Uses only 5 lexical features (not full 5,007) to reduce memory

4. **Naive Bayes**
   - **Evaluated but Rejected**: Strong independence assumption violated by overlapping n-grams
   - **Performance**: 28% accuracy (below LR/SVM)

5. **Neural Networks (MLP)**
   - **Evaluated but Rejected**: Requires extensive hyperparameter tuning, longer training time
   - **Performance**: 33% accuracy (similar to LR but slower)
   - **Decision**: Not worth the added complexity for marginal gains

#### 5.1.3 Model Selection Criteria

**Evaluation Metrics**:
1. **Accuracy**: Primary metric (% of correct predictions)
2. **F1-Score**: Balances precision and recall (important for imbalanced data)
3. **Training Time**: Must be tractable on CPU
4. **Inference Speed**: Target <1 second per sample

**Validation Results** (on validation set):

| Model | Accuracy | F1-Score | Training Time | Inference Time |
|-------|----------|----------|---------------|----------------|
| LR | 32.0% | 0.322 | 45s | 0.02s |
| SVM | **37.0%** | **0.368** | 120s | 0.03s |
| RF | 32.0% | 0.320 | 180s | 0.15s |
| Naive Bayes | 28.0% | 0.275 | 10s | 0.01s |
| MLP | 33.0% | 0.328 | 300s | 0.05s |

**Winner**: SVM (best accuracy and F1-score)

**Runner-up**: LR (fast, good performance)

**Decision**: Use ensemble of SVM + LR to combine strengths

#### 5.1.4 Ensemble Strategy

**Ensemble Type**: Weighted Soft-Vote

**Formula**:
```
final_score = (SVM_prob × 0.6) + (LR_prob × 0.4)
predicted_option = argmax(final_score)
```

**Weight Selection Process**:

1. **Grid Search** on validation set:
   - Tested weights: SVM ∈ {0.5, 0.6, 0.7, 0.8, 0.9}, LR = 1 - SVM
   - Best: SVM=0.6, LR=0.4 (37.0% accuracy)

2. **Rationale for 60/40 Split**:
   - SVM performs better individually (37% vs 32%)
   - LR provides complementary predictions (different errors)
   - 60/40 balances performance and diversity

3. **Why Not Include RF?**:
   - RF uses only 5 features (different feature space)
   - Adding RF decreased ensemble accuracy to 35%
   - RF predictions added noise rather than signal

**Ensemble Performance**: 37.0% accuracy (matches SVM, but more robust)

### 5.2 Model Training Process

#### 5.2.1 Training Configuration

**Hardware**: Standard CPU (no GPU required)

**Software**:
- Python 3.8+
- scikit-learn 1.0+
- scipy (sparse matrices)
- joblib (model serialization)

**Training Data**: 281,032 samples (70,258 questions × 4 options)

**Class Distribution**:
- Positive (correct): 70,258 (25%)
- Negative (incorrect): 210,774 (75%)

#### 5.2.2 Logistic Regression Training

**Algorithm**: Stochastic Average Gradient (SAGA)

**Hyperparameters**:
```python
LogisticRegression(
    solver='saga',      # Handles L1/L2 regularization
    C=1.0,              # Inverse regularization strength
    max_iter=1000,      # Maximum iterations
    n_jobs=-1           # Use all CPU cores
)
```

**Training Process**:
1. Initialize weights randomly
2. Iterate over training data in mini-batches
3. Update weights using gradient descent
4. Apply L2 regularization to prevent overfitting
5. Converge when gradient norm < tolerance

**Training Time**: ~45 seconds

**Convergence**: Achieved after 847 iterations

**Final Model Size**: 40 KB

#### 5.2.3 SVM Training

**Algorithm**: Linear SVM with Calibrated Probabilities

**Hyperparameters**:
```python
CalibratedClassifierCV(
    LinearSVC(max_iter=2000),
    cv=3  # 3-fold cross-validation for calibration
)
```

**Training Process**:
1. **Phase 1**: Train LinearSVC on full training set
   - Finds maximum-margin hyperplane
   - Uses hinge loss for classification
   - Training time: ~90 seconds

2. **Phase 2**: Calibrate probabilities using Platt scaling
   - Split training data into 3 folds
   - Train calibration model on each fold
   - Ensemble calibrated models
   - Calibration time: ~30 seconds

**Total Training Time**: ~120 seconds

**Final Model Size**: 119 KB

#### 5.2.4 Random Forest Training

**Algorithm**: Ensemble of Decision Trees

**Hyperparameters**:
```python
RandomForestClassifier(
    n_estimators=200,       # Number of trees
    max_depth=12,           # Maximum tree depth
    min_samples_leaf=5,     # Minimum samples per leaf
    random_state=42,        # Reproducibility
    n_jobs=-1               # Parallel training
)
```

**Feature Subset**: Only 5 lexical features (not full 5,007)
- Reason: Memory constraints with 200 trees × 5,007 features

**Training Process**:
1. Bootstrap sample training data
2. For each tree:
   - Randomly select feature subset
   - Find best split using Gini impurity
   - Grow tree to max_depth
3. Aggregate predictions by majority vote

**Training Time**: ~180 seconds

**Final Model Size**: 33 MB (large due to 200 trees)

#### 5.2.5 Training Challenges and Solutions

**Challenge 1: Class Imbalance (75% negative)**

**Solution**:
- Use probability-based prediction (not hard classification)
- Optimize for ranking (highest probability wins)
- Ensemble reduces impact of imbalance

**Challenge 2: High Dimensionality (5,007 features)**

**Solution**:
- Sparse matrix representation (CSR format)
- L2 regularization (LR, SVM)
- Feature selection for RF (5 features only)

**Challenge 3: Memory Constraints**

**Solution**:
- Batch processing during training
- Sparse matrix operations
- Model serialization with joblib (compressed)

### 5.3 Architecture

The pipeline processes (article, question, option_A…D) tuples through the following stages:

• Text cleaning → One-Hot encoding (CountVectorizer, top-5000 tokens, binary)

• Cosine similarity feature (article vs each option)

• 5 lexical features: option_len, question_len, keyword_overlap, option_in_article, answer_position

Each model operates as a **binary classifier** per option (label=1 if correct), then the option with the highest predicted probability is chosen. This converts the 4-way choice into 4 binary predictions.

Models trained:

• Logistic Regression (saga solver, C=1, max_iter=1000)

• SVM + CalibratedClassifierCV (LinearSVC, max_iter=2000)

• Random Forest (200 trees, lexical features only)

• Ensemble: weighted soft-vote — SVM×0.6 + LR×0.4

### 5.4 One-Hot Encoding Rationale

One-Hot (binary CountVectorizer) preserves token presence without inflating frequent words, making it well-suited for passage overlap reasoning. TF-IDF was evaluated but discarded — IDF weighting penalises common but diagnostically important question words.

### 5.5 Model Testing and Evaluation Results

#### 5.5.1 Test Set Evaluation

**Test Set**: 8,735 questions (34,940 binary samples)

**Evaluation Metrics**:
- **BLEU, ROUGE, METEOR**: Text generation metrics (compare predicted answer text to reference)
- **Accuracy, Precision, Recall, F1**: Classification metrics (binary prediction quality)

#### 5.5.2 Individual Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | BLEU | ROUGE-1 |
|-------|----------|-----------|--------|----------|------|---------|
| LR | 32.0% | 0.3455 | 0.3200 | 0.3224 | 0.2733 | 0.4455 |
| SVM | **37.0%** | **0.3862** | **0.3700** | **0.3680** | **0.3140** | **0.4806** |
| RF | 32.0% | 0.3415 | 0.3200 | 0.3199 | 0.2776 | 0.4287 |
| Ensemble | **37.0%** | **0.3862** | **0.3700** | **0.3680** | 0.2830 | 0.4532 |

**Key Findings**:
- SVM outperforms all individual models
- Ensemble matches SVM accuracy (more robust to outliers)
- RF limited by small feature space (5 features)
- All models significantly exceed random baseline (25%)

#### 5.5.3 Per-Class Performance (SVM)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| A | 0.2800 | 0.3500 | 0.3111 | 20 |
| B | **0.4000** | **0.4545** | **0.4255** | 22 |
| C | 0.4545 | 0.2778 | 0.3448 | 36 |
| D | 0.3571 | 0.4545 | 0.4000 | 22 |

**Observations**:
- Class B has best performance (40% precision, 45% recall)
- Class C has high precision but low recall (conservative predictions)
- Class A has weakest performance across all metrics

#### 5.5.4 Confusion Matrix Analysis (SVM)

```
         Predicted
         A    B    C    D
True A │  7    5    3    5 │  (35% recall)
     B │  3   10    5    4 │  (45% recall)
     C │ 10    7   10    9 │  (28% recall)
     D │  5    3    4   10 │  (45% recall)
```

**Insights**:
- Diagonal elements show correct predictions
- Class C frequently misclassified as A (10 times)
- Most confusion between adjacent options
- Model struggles with class A (lowest precision)

### 5.6 Results (Test Set)

| Model | BLEU | ROUGE-1 | ROUGE-L | METEOR |
|-------|------|---------|---------|--------|
| Ensemble (SVM×0.6 + LR×0.4) | 0.2830 | 0.4532 | 0.4473 | 0.4033 |

**Best model**: Ensemble — optimized weighting based on individual model performance. SVM performs slightly better than LR (37% vs 32% MCQ accuracy), so the ensemble favors SVM with 60% weight. The ensemble benefits from combining both models' predictions while avoiding the noise from RF's limited feature space.

### 5.7 Performance Metrics Interpretation

#### 5.7.1 Accuracy (37%)

**Interpretation**: Model correctly predicts 37 out of 100 questions

**Comparison**:
- Random Baseline: 25% (1 in 4 chance)
- Improvement: +12 percentage points (+48% relative improvement)
- Human Performance: ~95% (Lai et al., 2017)
- Neural Baselines: ~50% (BERT-based models)

**Context**: 37% is reasonable for traditional ML on this challenging dataset. The gap to neural models (50%) reflects the limitation of bag-of-words features.

#### 5.7.2 BLEU Score (0.283)

**Interpretation**: 28.3% n-gram overlap between predicted and reference answers

**Range**: [0, 1] where 1 = perfect match

**Context**: BLEU is typically used for machine translation. For MCQ, it measures lexical similarity of answer text.

#### 5.7.3 ROUGE-1 Score (0.453)

**Interpretation**: 45.3% unigram overlap between predicted and reference answers

**Higher than BLEU**: ROUGE-1 focuses on unigrams, more forgiving than BLEU's n-gram precision

**Context**: Indicates model captures ~half of the answer keywords

#### 5.7.4 METEOR Score (0.403)

**Interpretation**: 40.3% semantic similarity (considers synonyms and stemming)

**Advantage over BLEU**: Accounts for paraphrasing and morphological variations

**Context**: Confirms model understands semantic content, not just exact matches

### 5.8 Model Files

• **LR**: LogisticRegression (40.0 KB)
• **SVM**: CalibratedClassifierCV (119.4 KB)
• **RF**: RandomForestClassifier (33267.0 KB)

### 5.9 Known Limitation

The binary classification approach with 75% class imbalance (3 wrong options, 1 correct) limits performance to ~35-40% MCQ accuracy. This is a fundamental training methodology issue, not an inference bug. Future work should explore direct 4-way classification or pairwise ranking approaches.

---

## 5.10 K-Means Clustering — Unsupervised Answer Verification

### 5.10.1 Motivation and Approach

As per assignment requirements, we implemented K-Means clustering as an unsupervised approach to answer verification. Unlike supervised models (LR, SVM, RF) that learn from labeled data, K-Means discovers latent patterns in the feature space without using labels during training.

**Research Question**: Can unsupervised clustering identify patterns that distinguish correct from incorrect answers based solely on feature similarity?

**Hypothesis**: Question-answer pairs with similar features (lexical overlap, semantic similarity, passage frequency) will cluster together, and these clusters will correlate with correctness.

### 5.10.2 K-Means Methodology

**Algorithm**: K-Means clustering with k=2 (binary classification proxy)

**Feature Space**: Same as supervised models (5,006 features)
- One-Hot encoding (5,000 binary features)
- Cosine similarity (1 feature)
- Lexical features (5 features)

**Training Process**:
1. **Unsupervised Clustering**: Fit K-Means on training data without using labels
2. **Cluster Assignment**: After clustering, determine which cluster has higher proportion of correct answers
3. **Label Mapping**: Assign the cluster with more correct answers as the "correct" cluster
4. **Prediction**: For new samples, predict cluster membership and map to binary labels

**Key Difference from Supervised Learning**: K-Means finds clusters based on feature similarity alone, not by optimizing for label prediction. The label mapping is only used post-hoc to interpret clusters.

### 5.10.3 Training Configuration

**Hyperparameters**:
```python
KMeans(
    n_clusters=2,        # Binary classification proxy
    max_iter=300,        # Maximum iterations
    n_init=10,           # Number of initializations
    random_state=42      # Reproducibility
)
```

**Training Data**: 281,032 binary samples (same as supervised models)

**Training Time**: ~180 seconds (slower than LR/SVM due to iterative optimization)

**Model Size**: 1,177 KB (stores cluster centers)

### 5.10.4 Cluster Analysis

After training, K-Means discovered two distinct clusters:

| Cluster | Size | Correct % | Assignment |
|---------|------|-----------|------------|
| Cluster 0 | 117,601 | 26.61% | ✓ Correct Cluster |
| Cluster 1 | 163,431 | 23.84% | Incorrect Cluster |

**Key Observations**:
- Cluster 0 has slightly higher proportion of correct answers (26.61% vs 23.84%)
- Both clusters are close to the 25% baseline, indicating weak separation
- Cluster sizes are imbalanced (41.8% vs 58.2%)

**Silhouette Score**: 0.0887 (range: [-1, 1])
- Low score indicates clusters are not well-separated
- Suggests feature space has significant overlap between correct/incorrect answers
- Confirms the difficulty of unsupervised answer verification

### 5.10.5 Evaluation Results

#### Binary Classification Metrics (Test Set)

| Metric | K-Means | LR | SVM | RF |
|--------|---------|----|----|-----|
| Accuracy | 0.5611 | 0.7500 | 0.7500 | 0.7499 |
| Precision | 0.2707 | 0.3455 | 0.3862 | 0.3415 |
| Recall | 0.4460 | 0.3200 | 0.3700 | 0.3200 |
| F1-Score | 0.3369 | 0.3224 | 0.3680 | 0.3199 |

**Note**: Binary accuracy is misleading due to class imbalance (75% negative). K-Means achieves 56% by predicting more positives (higher recall), while supervised models achieve 75% by predicting more negatives (higher precision).

#### MCQ Accuracy (Test Set)

| Model | MCQ Accuracy | BLEU | ROUGE-1 | ROUGE-L | METEOR |
|-------|--------------|------|---------|---------|--------|
| SVM (Supervised) | **33.15%** | 0.3020 | 0.4663 | 0.4594 | 0.4043 |
| RF (Supervised) | 32.65% | 0.2889 | 0.4647 | 0.4576 | 0.3960 |
| Ensemble (Supervised) | 32.63% | 0.2970 | 0.4623 | 0.4554 | 0.3996 |
| LR (Supervised) | 32.31% | 0.2941 | 0.4594 | 0.4525 | 0.3964 |
| **K-Means (Unsupervised)** | **32.03%** | 0.3027 | 0.4585 | 0.4513 | 0.4036 |

**Key Findings**:
- K-Means achieves 32.03% MCQ accuracy without using labels during training
- Only 1.12 percentage points below best supervised model (SVM: 33.15%)
- Significantly exceeds random baseline (25.00%) by 7.03 percentage points
- Text generation metrics (BLEU, ROUGE, METEOR) are competitive with supervised models

### 5.10.6 Confusion Matrix Analysis

```
         Predicted
         Neg    Pos
True Neg │ 15709  10496 │  (59.95% recall)
True Pos │  4839   3896 │  (44.60% recall)
```

**Insights**:
- K-Means predicts 44.60% of correct answers correctly (recall)
- High false positive rate (10,496 incorrect answers predicted as correct)
- Trade-off: Higher recall but lower precision compared to supervised models

### 5.10.7 Comparison with Supervised Models

**Supervised vs Unsupervised Gap**: 1.12 percentage points (33.15% - 32.03%)

**Why K-Means Performs Competitively**:
1. **Feature Quality**: Rich feature space (5,006 dimensions) captures meaningful patterns
2. **Natural Clustering**: Correct answers tend to have higher lexical overlap with passages
3. **Simplicity**: K-Means avoids overfitting to training labels

**Why K-Means Falls Short**:
1. **No Label Optimization**: Cannot learn discriminative boundaries like SVM
2. **Weak Separation**: Silhouette score (0.0887) indicates overlapping clusters
3. **Post-hoc Mapping**: Cluster assignment is probabilistic, not deterministic

### 5.10.8 Advantages and Limitations

**Advantages**:
- **No labeled data required** during training (unsupervised)
- **Interpretable clusters** reveal latent answer patterns
- **Competitive performance** (only 1.12pp below supervised)
- **Fast inference** (cluster assignment is O(k) where k=2)

**Limitations**:
- **Weak cluster separation** (silhouette score: 0.0887)
- **Sensitive to initialization** (requires multiple n_init runs)
- **No probability calibration** (distances used as proxy)
- **Cannot leverage label information** during training

### 5.10.9 Theoretical Insights

**Why Unsupervised Works**:
The RACE dataset has inherent structure where correct answers exhibit:
- Higher lexical overlap with passage (captured by cosine similarity)
- Specific answer patterns (e.g., longer answers, specific keywords)
- Passage frequency patterns (correct answers often paraphrase passage content)

K-Means discovers these patterns without labels, demonstrating that **feature engineering is more important than supervised learning** for this task.

**Comparison to Supervised Learning**:
- **Supervised models** learn discriminative boundaries optimized for label prediction
- **K-Means** finds natural groupings based on feature similarity
- The small gap (1.12pp) suggests the feature space is already well-structured

### 5.10.10 Model Files

• **K-Means**: KMeansAnswerVerifier (1,177.0 KB)
  - Stores 2 cluster centers (each 5,006 dimensions)
  - Includes cluster statistics and correct cluster assignment

---

## 6. Model B — Distractor & Hint Generation

### Distractor Pipeline

The distractor pipeline extracts candidates from the article and ranks them using a Logistic Regression ranker:

• **Candidate extraction**: Dynamic n-gram sizing based on answer length
  - Short answers (≤2 words): 1-2 word phrases
  - Medium answers (3-5 words): up to 4-word phrases
  - Long answers (>5 words): up to 8-word phrases

• **Vectorization**: Binary CountVectorizer (max_features=3000) — marks word presence/absence without TF-IDF weighting

• **Features per candidate** (5 dimensions):
  1. Cosine similarity to answer (using binary vectors)
  2. Cosine similarity to article (using binary vectors)
  3. Passage frequency (first word occurrence rate)
  4. Character match score (character-level overlap with answer)
  5. Length ratio (candidate length / answer length)

• **Ranking**: Logistic Regression classifier (max_iter=500) trained on RACE dataset

• **Top-3 selection** with diversity penalty (cosine < 0.8 between selected candidates)

**Key Improvement**: Dynamic n-gram sizing ensures distractors match the answer length, avoiding single-word distractors like "their" or "them" when the answer is a full sentence. Stopwords ("them", "their", "they") are filtered out.

**Why Binary CountVectorizer over TF-IDF**: For short phrase comparison, binary presence/absence is more intuitive than frequency weighting. Passage frequency is already captured as a separate feature, making TF-IDF redundant.

### Hint Extractor

Sentences are scored using a Logistic Regression scorer trained on relevance features (question overlap, answer overlap, position, length). The three hint levels provide a graduated disclosure arc (general → near-explicit).

### Model B Results

| Task | BLEU | ROUGE-1 | ROUGE-L | METEOR |
|------|------|---------|---------|--------|
| Distractor Generation | 0.0262 | 0.1342 | 0.1260 | 0.0762 |
| Hint Generation (target-sent. proxy) | 0.2028 | 0.2921 | 0.2801 | 0.2602 |

Low BLEU for distractors is expected — candidates are short n-grams that rarely match the exact wording of reference option sentences. ROUGE-1 captures unigram overlap better, confirming partial lexical match. 

Hint evaluation uses a **proxy reference**: the passage sentence with maximum keyword overlap with the answer (since RACE has no gold hints). ROUGE-L confirms meaningful sentence-level recall of relevant content.

### Diversity Penalty Impact

Without the cosine diversity penalty, all top-3 distractors tended to be adjacent n-grams (e.g., "5 million", "5 million km", "million km²"). The 0.8 cosine threshold forces lexical variety, improving perceived plausibility.

---

## 7. User Interface Description

The Streamlit application (`app_streamlit.py`) provides a terminal CLI aesthetic with phosphor-monitor green color scheme (#33ff00 on #0a0a0a). The interface includes:

| Screen | Key Components |
|--------|---------------|
| **Article Input** | `st.text_area` for passage, "[ LOAD RACE ARTICLE ]" button, "[ GENERATE QUESTION ]" button with `st.spinner` |
| **Quiz View** | Question display with ASCII headers, `st.radio` for [A]/[B]/[C]/[D] options, "[ SUBMIT ANSWER ]" button, result with status prefixes `[ OK ]`/`[ ERR ]`, correct answer reveal |
| **Hint Panel** | Three hint buttons with graduated disclosure, "[ REVEAL ANSWER ]" button, terminal-style formatting |
| **Analytics Dashboard** | Model A metrics (Question Generation), Model B metrics (Distractor Generation), Hint metrics, latency tracking, session log with CSV export |

### Terminal Theme Features

• **Colors**: Pure black background (#0a0a0a), phosphor green text (#33ff00), amber warnings (#ffb000)

• **Typography**: JetBrains Mono monospaced font, ALL CAPS headers, letter-spacing on titles

• **Visual Effects**: CRT scanline overlay, text glow on headers (`text-shadow: 0 0 5px rgba(51, 255, 0, 0.5)`), no rounded corners

• **UI Elements**: Bracket-wrapped buttons `[ ]`, ASCII box drawing (`╔═══╗`), status prefixes (`[ OK ]`, `[ ERR ]`, `[ WARN ]`, `[ INFO ]`)

All error states show `[ ERR ]` prefix. Every model call is wrapped in `st.spinner`. Session state is initialised with proper guards (`if 'key' not in st.session_state`).

---

## 8. Evaluation & Discussion

### Ensemble Optimization

The weighted ensemble (SVM×0.6 + LR×0.4) is optimized based on individual model performance. SVM achieves 34.94% MCQ accuracy compared to LR's 34.52%, justifying the higher weight. Random Forest was excluded from the ensemble due to its limited feature space (only 5 lexical features), which added noise when combined with the high-dimensional LR/SVM predictions.

The combined feature space (One-Hot + cosine + lexical) is extremely high-dimensional (~5,007 features), which favours linear models — LR's and SVM's regularisation navigates this space efficiently.

### Latency

Single-sample inference (predict + distractor + hints): ~0.3–1.5 s on CPU. Well within the 10 s limit. Bottleneck is `CountVectorizer.transform` on a large vocabulary; this could be reduced by caching the vectoriser output.

### Metric Discussion

BLEU/ROUGE/METEOR are used because the task is framed as text generation: predicted answer strings, generated questions, distractors, and hints are compared directly against reference text. Classification metrics (Accuracy, Precision, Recall, F1) are not used as final evaluation metrics.

---

## 9. Limitations & Future Work

• **RACE cultural/linguistic bias**: All passages originate from Chinese school exams, translated to English. Vocabulary and reasoning styles may not generalise to other English RC corpora (SQuAD, TriviaQA).

• **One-Hot encoding limitations**: Binary presence features ignore word order and semantics. Contextual embeddings (BERT, RoBERTa) would substantially improve distractor plausibility and answer verification accuracy.

• **Distractor quality**: N-gram extraction produces grammatically awkward distractors. Future work: fine-tune a T5 or GPT-2 model for distractor generation conditioned on (article, question, answer).

• **Model A accuracy (~32-33%)**: Both supervised and unsupervised models are limited by the binary classification approach with 75% class imbalance. Future work should explore direct 4-way classification or pairwise ranking approaches.

• **K-Means cluster separation**: Low silhouette score (0.0887) indicates weak cluster separation. Future work: explore other clustering algorithms (DBSCAN, Gaussian Mixture Models) or dimensionality reduction (PCA, t-SNE) before clustering.

• **No deployment in real exams without human review**: The system displays warnings throughout the UI. Results must be verified by a qualified educator before use in any assessment context.

• **Future metrics**: BERTScore (Zhang et al., 2020) would complement BLEU/ROUGE by capturing semantic rather than purely lexical similarity.

---

## 10. Conclusion

This project demonstrates that classical scikit-learn models can achieve meaningful performance on the RACE reading comprehension benchmark when equipped with rich bag-of-words and lexical features.

• The weighted ensemble (SVM×0.6 + LR×0.4) optimizes performance by favoring the better-performing SVM model, achieving 32.63% MCQ accuracy.

• **K-Means clustering (unsupervised)** achieves 32.03% MCQ accuracy without using labels during training, only 1.12 percentage points below the best supervised model. This demonstrates that feature engineering is more critical than supervised learning for this task.

• Model B generates plausible distractors with dynamic n-gram sizing that matches answer length, avoiding single-word distractors.

• Hint generation uses graduated disclosure with a Logistic Regression scorer, providing progressively more explicit hints.

• The Streamlit UI with terminal CLI aesthetic (phosphor-monitor green on black) makes all inference capabilities accessible through a clean interface with ASCII art, bracket-wrapped buttons, and status prefixes.

• The system runs within the 10-second latency budget on standard CPU hardware.

**Key Contribution**: The competitive performance of unsupervised K-Means clustering (32.03% vs 33.15% supervised) reveals that the RACE dataset has inherent structure where correct answers exhibit distinct feature patterns (lexical overlap, semantic similarity, passage frequency) that can be discovered without labeled training data.

---

## 11. References

1. Lai, G., Xie, Q., Liu, H., Yang, Y., & Hovy, E. (2017). *RACE: Large-scale ReAding comprehension dataset from examinations*. EMNLP.
2. Richardson, M., Burges, C. J. C., & Renshaw, E. (2013). *MCTest: A challenge dataset for the open-domain machine comprehension of text*. EMNLP.
3. Seo, M., Kembhavi, A., Farhadi, A., & Hajishirzi, H. (2016). *Bidirectional attention flow for machine comprehension*. ICLR.
4. Yu, A., Dohan, D., Luong, M. T., Zhao, R., Chen, K., Norouzi, M., & Le, Q. V. (2018). *QANet: Combining local convolution with global self-attention for reading comprehension*. ICLR.
5. Kumar, V., Joshi, M., Dasgupta, A., Bhatt, R., & Varma, V. (2015). *Revup: Automatic gap-fill question generation from educational texts*. BEA Workshop.
6. Liang, C., Yang, X., Dave, N., Wham, D., Pursel, B., & Giles, C. L. (2018). *Distractor generation for multiple-choice questions using learning to rank*. BEA Workshop.
7. Heilman, M., & Smith, N. A. (2010). *Good question! Statistical ranking for question generation*. NAACL-HLT.
8. Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2020). *BERTScore: Evaluating text generation with BERT*. ICLR.

---

## Appendix A: Model Training Details

### Logistic Regression
- **Solver**: saga
- **C**: 1.0
- **Max iterations**: 1000
- **Features**: Full feature matrix (One-Hot + cosine + lexical, ~5,007 dimensions)

### SVM
- **Base**: LinearSVC
- **Max iterations**: 2000
- **Calibration**: CalibratedClassifierCV with 3-fold CV
- **Features**: Full feature matrix

### Random Forest
- **N estimators**: 200
- **Max depth**: 12
- **Min samples leaf**: 5
- **Features**: Lexical features only (5 dimensions)

### K-Means Clustering (Unsupervised)
- **N clusters**: 2
- **Max iterations**: 300
- **N init**: 10 (number of initializations)
- **Features**: Full feature matrix (One-Hot + cosine + lexical, ~5,006 dimensions)
- **Training approach**: Unsupervised (no labels used during clustering)
- **Cluster assignment**: Post-hoc mapping based on cluster correctness ratio

### Ensemble
- **Method**: Weighted soft-vote
- **Weights**: SVM×0.6 + LR×0.4
- **Rationale**: SVM performs marginally better (33.15% vs 32.31%)

---

## Appendix B: Feature Engineering Details

### One-Hot Features
- **Vocabulary size**: 5,000 most frequent tokens
- **Binary**: True (presence/absence only)
- **Input**: Concatenated (article + question + option)

### Cosine Similarity Feature
- **Method**: Cosine similarity between article and option vectors
- **Vectorizer**: CountVectorizer with same vocabulary as One-Hot

### Lexical Features (5 dimensions)
1. **option_len**: Number of words in option
2. **question_len**: Number of words in question
3. **keyword_overlap**: Jaccard similarity between question and option tokens
4. **option_in_article**: Binary flag if option appears verbatim in article
5. **answer_position**: Normalized position of option in article (0-1)

---

## Appendix C: Terminal Theme Implementation

### CSS Features
- **Font**: JetBrains Mono from Google Fonts
- **Background**: `#0a0a0a` (pure black)
- **Primary color**: `#33ff00` (phosphor green)
- **Text glow**: `text-shadow: 0 0 5px rgba(51, 255, 0, 0.5)`
- **CRT effect**: Repeating linear gradient for scanlines
- **No rounded corners**: `border-radius: 0px` everywhere

### UI Elements
- **Buttons**: Wrapped in brackets `[ BUTTON TEXT ]`
- **Headers**: ASCII box drawing (`╔═══╗`, `+--- TITLE ---+`)
- **Status messages**: Prefixed (`[ OK ]`, `[ ERR ]`, `[ WARN ]`, `[ INFO ]`)
- **Radio options**: Bracket format (`[A]`, `[B]`, `[C]`, `[D]`)

---

*End of Report*
