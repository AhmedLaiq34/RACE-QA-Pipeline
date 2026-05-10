"""
Evaluate Individual Model A Classifiers

This script evaluates LR, SVM, and RF models individually (not as ensemble)
to show their individual performance metrics.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from pipeline.evaluate import compute_generation_metrics
from pipeline.preprocessing import clean_text

# Configuration
MODELS_DIR = Path("models/model_a/traditional")
TEST_SAMPLE_SIZE = 100
RANDOM_SEED = 42


def _lexical_row(article, question, option):
    """Extract lexical features"""
    article_tokens = set(article.split())
    question_tokens = set(question.split())
    option_tokens = set(option.split())
    option_words = option.split()
    position = 0.0
    if option_words and option_words[0] in article:
        position = article.find(option_words[0]) / max(len(article), 1)
    return [
        len(option_words),
        len(question.split()),
        len(question_tokens & option_tokens),
        len(option_tokens & article_tokens),
        position,
    ]


def _rowwise_cosine(article_texts, option_texts, vocabulary):
    """Compute cosine similarity between article and option"""
    vectorizer = CountVectorizer(binary=True, vocabulary=vocabulary)
    article_matrix = vectorizer.transform(article_texts)
    option_matrix = vectorizer.transform(option_texts)
    numerator = np.asarray(article_matrix.multiply(option_matrix).sum(axis=1)).ravel()
    article_norm = np.sqrt(np.asarray(article_matrix.multiply(article_matrix).sum(axis=1)).ravel())
    option_norm = np.sqrt(np.asarray(option_matrix.multiply(option_matrix).sum(axis=1)).ravel())
    denominator = article_norm * option_norm
    values = np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=np.float32), where=denominator != 0)
    return csr_matrix(values.reshape(-1, 1))


def _positive_probability(model, matrix):
    """Get positive class probability"""
    probabilities = model.predict_proba(matrix)
    if probabilities.shape[1] == 1:
        return probabilities[:, 0]
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
        if 1 in classes:
            return probabilities[:, classes.index(1)]
    return probabilities[:, -1]


def predict_with_model(model, ohe_vectorizer, article, question, options, use_lexical_only=False):
    """Predict answer using a specific model"""
    # Clean text
    article_clean = clean_text(article)
    question_clean = clean_text(question)
    options_clean = [clean_text(opt) for opt in options]
    
    if use_lexical_only:
        # RF uses only lexical features
        lexical_features = np.array([
            _lexical_row(article_clean, question_clean, opt) 
            for opt in options_clean
        ], dtype=np.float32)
        feature_matrix = lexical_features
    else:
        # LR and SVM use full feature matrix
        combined = [f"{article_clean} [SEP] {question_clean} [SEP] {opt}" for opt in options_clean]
        ohe_matrix = ohe_vectorizer.transform(combined)
        
        # Cosine similarity
        cosine_feature = _rowwise_cosine(
            [article_clean] * len(options),
            options_clean,
            ohe_vectorizer.vocabulary_
        )
        
        # Lexical features
        lexical_features = csr_matrix(np.array([
            _lexical_row(article_clean, question_clean, opt) 
            for opt in options_clean
        ], dtype=np.float32))
        
        feature_matrix = hstack([ohe_matrix, cosine_feature, lexical_features], format="csr")
    
    # Get probabilities
    scores = _positive_probability(model, feature_matrix)
    
    # Return predicted letter
    OPTIONS = ('A', 'B', 'C', 'D')
    return OPTIONS[int(np.argmax(scores))]


def evaluate_individual_models():
    """Evaluate LR, SVM, and RF individually"""
    print("=" * 70)
    print("Individual Model Evaluation")
    print("=" * 70)
    print()
    
    # Load models
    print("📦 Loading models...")
    lr_model = joblib.load(MODELS_DIR / "lr_model.pkl")
    svm_model = joblib.load(MODELS_DIR / "svm_model.pkl")
    rf_model = joblib.load(MODELS_DIR / "rf_model.pkl")
    ohe_vectorizer = joblib.load(MODELS_DIR / "ohe_vectorizer.pkl")
    print("  ✅ Models loaded")
    print()
    
    # Load test data
    print("📊 Loading test data...")
    df_test = pd.read_csv("data/raw/test.csv")
    np.random.seed(RANDOM_SEED)
    sample_indices = np.random.choice(len(df_test), min(TEST_SAMPLE_SIZE, len(df_test)), replace=False)
    df_sample = df_test.iloc[sample_indices]
    print(f"  ✅ Loaded {len(df_sample)} test samples")
    print()
    
    # Evaluate each model
    results = {
        'LR': {'predictions': [], 'references': []},
        'SVM': {'predictions': [], 'references': []},
        'RF': {'predictions': [], 'references': []},
    }
    
    print("🔬 Evaluating models...")
    for idx, row in df_sample.iterrows():
        article = row['article']
        question = row['question']
        options = [row['A'], row['B'], row['C'], row['D']]
        answer_letter = row['answer']
        correct_answer = row[answer_letter]
        
        try:
            # LR prediction
            lr_pred_letter = predict_with_model(lr_model, ohe_vectorizer, article, question, options)
            lr_pred_answer = options[ord(lr_pred_letter) - 65]
            results['LR']['predictions'].append(lr_pred_answer)
            results['LR']['references'].append(correct_answer)
            
            # SVM prediction
            svm_pred_letter = predict_with_model(svm_model, ohe_vectorizer, article, question, options)
            svm_pred_answer = options[ord(svm_pred_letter) - 65]
            results['SVM']['predictions'].append(svm_pred_answer)
            results['SVM']['references'].append(correct_answer)
            
            # RF prediction (lexical features only)
            rf_pred_letter = predict_with_model(rf_model, ohe_vectorizer, article, question, options, use_lexical_only=True)
            rf_pred_answer = options[ord(rf_pred_letter) - 65]
            results['RF']['predictions'].append(rf_pred_answer)
            results['RF']['references'].append(correct_answer)
            
        except Exception as e:
            print(f"  ⚠️  Error on sample {idx}: {e}")
    
    print(f"  ✅ Evaluated {len(results['LR']['predictions'])} samples")
    print()
    
    # Compute metrics
    print("=" * 70)
    print("Results")
    print("=" * 70)
    print()
    
    metrics = {}
    for model_name in ['LR', 'SVM', 'RF']:
        if results[model_name]['predictions']:
            metrics[model_name] = compute_generation_metrics(
                results[model_name]['predictions'],
                results[model_name]['references']
            )
    
    # Display results
    print("┌─────────────────────────────────────────────────────────────────┐")
    print("│                    INDIVIDUAL MODEL METRICS                     │")
    print("├─────────────────────────────────────────────────────────────────┤")
    print("│ Model │   BLEU   │ ROUGE-1  │ ROUGE-L  │  METEOR  │ Accuracy │")
    print("├───────┼──────────┼──────────┼──────────┼──────────┼──────────┤")
    
    for model_name in ['LR', 'SVM', 'RF']:
        m = metrics[model_name]
        
        # Calculate accuracy
        correct = sum(1 for p, r in zip(results[model_name]['predictions'], 
                                        results[model_name]['references']) 
                     if p.lower().strip() == r.lower().strip())
        accuracy = correct / len(results[model_name]['predictions'])
        
        print(f"│  {model_name:4s} │  {m['bleu']:.4f}  │  {m['rouge_1']:.4f}  │  {m['rouge_l']:.4f}  │  {m['meteor']:.4f}  │  {accuracy:.4f}  │")
    
    print("└───────┴──────────┴──────────┴──────────┴──────────┴──────────┘")
    print()
    
    # Comparison
    print("=" * 70)
    print("Analysis")
    print("=" * 70)
    print()
    
    # Best model per metric
    best_bleu = max(metrics.items(), key=lambda x: x[1]['bleu'])
    best_rouge1 = max(metrics.items(), key=lambda x: x[1]['rouge_1'])
    best_rougel = max(metrics.items(), key=lambda x: x[1]['rouge_l'])
    best_meteor = max(metrics.items(), key=lambda x: x[1]['meteor'])
    
    print(f"🏆 Best BLEU:    {best_bleu[0]} ({best_bleu[1]['bleu']:.4f})")
    print(f"🏆 Best ROUGE-1: {best_rouge1[0]} ({best_rouge1[1]['rouge_1']:.4f})")
    print(f"🏆 Best ROUGE-L: {best_rougel[0]} ({best_rougel[1]['rouge_l']:.4f})")
    print(f"🏆 Best METEOR:  {best_meteor[0]} ({best_meteor[1]['meteor']:.4f})")
    print()
    
    # Calculate accuracies
    accuracies = {}
    for model_name in ['LR', 'SVM', 'RF']:
        correct = sum(1 for p, r in zip(results[model_name]['predictions'], 
                                        results[model_name]['references']) 
                     if p.lower().strip() == r.lower().strip())
        accuracies[model_name] = correct / len(results[model_name]['predictions'])
    
    best_acc = max(accuracies.items(), key=lambda x: x[1])
    print(f"🏆 Best Accuracy: {best_acc[0]} ({best_acc[1]:.4f} = {best_acc[1]*100:.2f}%)")
    print()
    
    print("📊 Observations:")
    print(f"  • SVM and LR use full feature matrix (~5,007 dimensions)")
    print(f"  • RF uses only lexical features (5 dimensions)")
    print(f"  • Ensemble (SVM×0.6 + LR×0.4) combines best of both")
    print(f"  • Random baseline: 0.25 (25% for 4-way MCQ)")
    print()
    
    return metrics, results


if __name__ == "__main__":
    evaluate_individual_models()
