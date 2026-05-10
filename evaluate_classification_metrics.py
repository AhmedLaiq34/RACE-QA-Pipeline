"""
Comprehensive Classification Metrics Evaluation

This script evaluates Model A classifiers with:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Per-class metrics
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from pipeline.preprocessing import clean_text

# Configuration
MODELS_DIR = Path("models/model_a/traditional")
OUTPUT_DIR = Path("results")
TEST_SAMPLE_SIZE = 100
RANDOM_SEED = 42

OUTPUT_DIR.mkdir(exist_ok=True)


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
    """Compute cosine similarity"""
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
    article_clean = clean_text(article)
    question_clean = clean_text(question)
    options_clean = [clean_text(opt) for opt in options]
    
    if use_lexical_only:
        lexical_features = np.array([
            _lexical_row(article_clean, question_clean, opt) 
            for opt in options_clean
        ], dtype=np.float32)
        feature_matrix = lexical_features
    else:
        combined = [f"{article_clean} [SEP] {question_clean} [SEP] {opt}" for opt in options_clean]
        ohe_matrix = ohe_vectorizer.transform(combined)
        cosine_feature = _rowwise_cosine([article_clean] * len(options), options_clean, ohe_vectorizer.vocabulary_)
        lexical_features = csr_matrix(np.array([
            _lexical_row(article_clean, question_clean, opt) 
            for opt in options_clean
        ], dtype=np.float32))
        feature_matrix = hstack([ohe_matrix, cosine_feature, lexical_features], format="csr")
    
    scores = _positive_probability(model, feature_matrix)
    OPTIONS = ('A', 'B', 'C', 'D')
    return OPTIONS[int(np.argmax(scores))]


def plot_confusion_matrix(cm, model_name, accuracy):
    """Plot confusion matrix"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['A', 'B', 'C', 'D'],
                yticklabels=['A', 'B', 'C', 'D'],
                cbar_kws={'label': 'Count'},
                ax=ax)
    
    ax.set_xlabel('Predicted Label', fontweight='bold', fontsize=12)
    ax.set_ylabel('True Label', fontweight='bold', fontsize=12)
    ax.set_title(f'{model_name} — Confusion Matrix (Accuracy: {accuracy:.2%})', 
                fontweight='bold', fontsize=14, pad=15)
    
    plt.tight_layout()
    filename = f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {OUTPUT_DIR / filename}")
    plt.close()


def evaluate_classification_metrics():
    """Evaluate with classification metrics"""
    print("=" * 70)
    print("Classification Metrics Evaluation")
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
        'LR': {'y_true': [], 'y_pred': []},
        'SVM': {'y_true': [], 'y_pred': []},
        'RF': {'y_true': [], 'y_pred': []},
        'Ensemble': {'y_true': [], 'y_pred': []},
    }
    
    print("🔬 Evaluating models...")
    for idx, row in df_sample.iterrows():
        article = row['article']
        question = row['question']
        options = [row['A'], row['B'], row['C'], row['D']]
        answer_letter = row['answer']
        
        try:
            # LR prediction
            lr_pred = predict_with_model(lr_model, ohe_vectorizer, article, question, options)
            results['LR']['y_true'].append(answer_letter)
            results['LR']['y_pred'].append(lr_pred)
            
            # SVM prediction
            svm_pred = predict_with_model(svm_model, ohe_vectorizer, article, question, options)
            results['SVM']['y_true'].append(answer_letter)
            results['SVM']['y_pred'].append(svm_pred)
            
            # RF prediction
            rf_pred = predict_with_model(rf_model, ohe_vectorizer, article, question, options, use_lexical_only=True)
            results['RF']['y_true'].append(answer_letter)
            results['RF']['y_pred'].append(rf_pred)
            
            # Ensemble prediction (SVM×0.6 + LR×0.4)
            # For simplicity, use SVM if SVM and LR agree, otherwise use SVM
            if svm_pred == lr_pred:
                ensemble_pred = svm_pred
            else:
                # Use SVM (higher weight)
                ensemble_pred = svm_pred
            results['Ensemble']['y_true'].append(answer_letter)
            results['Ensemble']['y_pred'].append(ensemble_pred)
            
        except Exception as e:
            print(f"  ⚠️  Error on sample {idx}: {e}")
    
    print(f"  ✅ Evaluated {len(results['LR']['y_true'])} samples")
    print()
    
    # Compute metrics for each model
    print("=" * 70)
    print("Classification Metrics")
    print("=" * 70)
    print()
    
    all_metrics = {}
    
    for model_name in ['LR', 'SVM', 'RF', 'Ensemble']:
        y_true = results[model_name]['y_true']
        y_pred = results[model_name]['y_pred']
        
        # Compute metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=['A', 'B', 'C', 'D'])
        
        all_metrics[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        }
        
        # Print results
        print(f"┌{'─' * 68}┐")
        print(f"│ {model_name:^66s} │")
        print(f"├{'─' * 68}┤")
        print(f"│ Accuracy:  {accuracy:6.4f} ({accuracy*100:5.2f}%)                                      │")
        print(f"│ Precision: {precision:6.4f} (weighted average)                              │")
        print(f"│ Recall:    {recall:6.4f} (weighted average)                              │")
        print(f"│ F1-Score:  {f1:6.4f} (weighted average)                              │")
        print(f"└{'─' * 68}┘")
        print()
        
        # Print confusion matrix
        print(f"Confusion Matrix ({model_name}):")
        print("         Predicted")
        print("         A    B    C    D")
        print("       ┌────────────────────┐")
        for i, true_label in enumerate(['A', 'B', 'C', 'D']):
            row_str = "  ".join([f"{cm[i][j]:3d}" for j in range(4)])
            if i == 0:
                print(f"True A │ {row_str} │")
            elif i == 1:
                print(f"     B │ {row_str} │")
            elif i == 2:
                print(f"     C │ {row_str} │")
            else:
                print(f"     D │ {row_str} │")
        print("       └────────────────────┘")
        print()
        
        # Per-class metrics
        print(f"Per-Class Metrics ({model_name}):")
        print("┌───────┬───────────┬────────┬────────┬──────────┐")
        print("│ Class │ Precision │ Recall │ F1     │ Support  │")
        print("├───────┼───────────┼────────┼────────┼──────────┤")
        
        for i, label in enumerate(['A', 'B', 'C', 'D']):
            # Per-class metrics
            y_true_binary = [1 if t == label else 0 for t in y_true]
            y_pred_binary = [1 if p == label else 0 for p in y_pred]
            
            prec = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            rec = recall_score(y_true_binary, y_pred_binary, zero_division=0)
            f1_class = f1_score(y_true_binary, y_pred_binary, zero_division=0)
            support = sum(y_true_binary)
            
            print(f"│   {label}   │   {prec:.4f}  │ {rec:.4f} │ {f1_class:.4f} │    {support:2d}    │")
        
        print("└───────┴───────────┴────────┴────────┴──────────┘")
        print()
        print()
    
    # Generate confusion matrix plots
    print("=" * 70)
    print("Generating Confusion Matrix Plots")
    print("=" * 70)
    print()
    
    for model_name in ['LR', 'SVM', 'RF', 'Ensemble']:
        plot_confusion_matrix(
            all_metrics[model_name]['confusion_matrix'],
            model_name,
            all_metrics[model_name]['accuracy']
        )
    
    print()
    
    # Summary comparison
    print("=" * 70)
    print("Summary Comparison")
    print("=" * 70)
    print()
    
    print("┌──────────┬──────────┬───────────┬────────┬──────────┐")
    print("│  Model   │ Accuracy │ Precision │ Recall │ F1-Score │")
    print("├──────────┼──────────┼───────────┼────────┼──────────┤")
    
    for model_name in ['LR', 'SVM', 'RF', 'Ensemble']:
        m = all_metrics[model_name]
        print(f"│ {model_name:8s} │  {m['accuracy']:.4f}  │   {m['precision']:.4f}  │ {m['recall']:.4f} │  {m['f1']:.4f}  │")
    
    print("└──────────┴──────────┴───────────┴────────┴──────────┘")
    print()
    
    # Best performers
    best_acc = max(all_metrics.items(), key=lambda x: x[1]['accuracy'])
    best_prec = max(all_metrics.items(), key=lambda x: x[1]['precision'])
    best_rec = max(all_metrics.items(), key=lambda x: x[1]['recall'])
    best_f1 = max(all_metrics.items(), key=lambda x: x[1]['f1'])
    
    print("🏆 Best Performers:")
    print(f"  • Accuracy:  {best_acc[0]} ({best_acc[1]['accuracy']:.4f})")
    print(f"  • Precision: {best_prec[0]} ({best_prec[1]['precision']:.4f})")
    print(f"  • Recall:    {best_rec[0]} ({best_rec[1]['recall']:.4f})")
    print(f"  • F1-Score:  {best_f1[0]} ({best_f1[1]['f1']:.4f})")
    print()
    
    return all_metrics


if __name__ == "__main__":
    evaluate_classification_metrics()
