"""Comprehensive evaluation of K-Means clustering for answer verification.

This script trains and evaluates K-Means clustering as an unsupervised approach
to answer verification, comparing it with supervised models (LR, SVM, RF).

The evaluation includes:
1. Binary classification metrics (accuracy, precision, recall, F1)
2. MCQ accuracy (selecting correct answer from 4 options)
3. Text generation metrics (BLEU, ROUGE, METEOR)
4. Confusion matrix visualization
5. Comparison with supervised baselines
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import load_npz

# Add pipeline to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pipeline"))

from kmeans_unsupervised import (
    train_kmeans_model,
    evaluate_kmeans_model,
    mcq_accuracy_from_binary,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Text generation metrics
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    from nltk.translate.meteor_score import meteor_score
    import nltk
    
    # Download required NLTK data
    try:
        nltk.data.find('wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    try:
        nltk.data.find('omw-1.4')
    except LookupError:
        nltk.download('omw-1.4', quiet=True)
    
    METRICS_AVAILABLE = True
except ImportError:
    print("Warning: NLTK or rouge-score not installed. Text generation metrics will be skipped.")
    METRICS_AVAILABLE = False


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models", "model_a", "traditional")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def calculate_mcq_accuracy(model, X, y, n_questions):
    """Calculate MCQ accuracy for a model.
    
    Args:
        model: Model with predict method
        X: Feature matrix
        y: True labels
        n_questions: Number of MCQ questions
    
    Returns:
        float: MCQ accuracy
    """
    y_pred = model.predict(X)
    
    correct = 0
    for i in range(n_questions):
        start_idx = i * 4
        end_idx = start_idx + 4
        
        # Get predictions for 4 options
        option_preds = y_pred[start_idx:end_idx]
        option_labels = y[start_idx:end_idx]
        
        # Find predicted and true options
        predicted_option = np.argmax(option_preds)
        true_option = np.argmax(option_labels)
        
        if predicted_option == true_option:
            correct += 1
    
    return correct / n_questions


def calculate_text_metrics(predicted_answers, reference_answers):
    """Calculate BLEU, ROUGE, and METEOR scores.
    
    Args:
        predicted_answers: List of predicted answer strings
        reference_answers: List of reference answer strings
    
    Returns:
        dict: Metrics dictionary
    """
    if not METRICS_AVAILABLE:
        return {
            'bleu': 0.0,
            'rouge_1': 0.0,
            'rouge_l': 0.0,
            'meteor': 0.0,
        }
    
    bleu_scores = []
    rouge_1_scores = []
    rouge_l_scores = []
    meteor_scores = []
    
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    smoothing = SmoothingFunction().method1
    
    for pred, ref in zip(predicted_answers, reference_answers):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        
        # BLEU
        if len(pred_tokens) > 0 and len(ref_tokens) > 0:
            bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing)
            bleu_scores.append(bleu)
        
        # ROUGE
        rouge_scores = rouge_scorer_obj.score(ref, pred)
        rouge_1_scores.append(rouge_scores['rouge1'].fmeasure)
        rouge_l_scores.append(rouge_scores['rougeL'].fmeasure)
        
        # METEOR
        try:
            meteor = meteor_score([ref_tokens], pred_tokens)
            meteor_scores.append(meteor)
        except:
            meteor_scores.append(0.0)
    
    return {
        'bleu': np.mean(bleu_scores) if bleu_scores else 0.0,
        'rouge_1': np.mean(rouge_1_scores),
        'rouge_l': np.mean(rouge_l_scores),
        'meteor': np.mean(meteor_scores),
    }


def evaluate_on_mcq_format(model, test_csv_path, model_name="K-Means"):
    """Evaluate model on MCQ format and calculate text metrics.
    
    Args:
        model: Trained model
        test_csv_path: Path to test CSV file
        model_name: Name of the model for display
    
    Returns:
        dict: Evaluation results
    """
    print(f"\n{'=' * 80}")
    print(f"EVALUATING {model_name} ON MCQ FORMAT")
    print('=' * 80)
    
    # Load test data
    test_df = pd.read_csv(test_csv_path)
    print(f"\nLoaded {len(test_df)} questions from {test_csv_path}")
    
    # Load vectorizer
    vectorizer = joblib.load(os.path.join(MODELS_DIR, "ohe_vectorizer.pkl"))
    
    # Prepare for evaluation
    from pipeline.preprocessing import clean_text, expand_df, prepare_text_columns
    
    test_prepared = prepare_text_columns(test_df)
    test_expanded = expand_df(test_prepared)
    
    # Extract features (simplified - just OHE for now)
    combined_texts = test_expanded["combined_text"]
    X_test_ohe = vectorizer.transform(combined_texts)
    
    # For full feature matrix, load from processed
    X_test_full = load_npz(os.path.join(PROCESSED_DIR, "X_test.npz"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
    
    # Predict
    y_pred = model.predict(X_test_full)
    
    # Calculate MCQ accuracy
    n_questions = len(test_df)
    mcq_acc = calculate_mcq_accuracy(model, X_test_full, y_test, n_questions)
    
    print(f"\nMCQ Accuracy: {mcq_acc:.4f} ({mcq_acc * 100:.2f}%)")
    print(f"Random Baseline: 0.2500 (25.00%)")
    print(f"Improvement: +{(mcq_acc - 0.25) * 100:.2f} percentage points")
    
    # Calculate text generation metrics
    predicted_answers = []
    reference_answers = []
    
    for i in range(n_questions):
        start_idx = i * 4
        end_idx = start_idx + 4
        
        # Get predictions for 4 options
        option_preds = y_pred[start_idx:end_idx]
        predicted_option_idx = np.argmax(option_preds)
        
        # Get answer texts
        row = test_df.iloc[i]
        options = ['A', 'B', 'C', 'D']
        predicted_answer = row[options[predicted_option_idx]]
        reference_answer = row[row['answer']]
        
        predicted_answers.append(str(predicted_answer))
        reference_answers.append(str(reference_answer))
    
    # Calculate text metrics
    text_metrics = calculate_text_metrics(predicted_answers, reference_answers)
    
    print(f"\n{'-' * 80}")
    print("TEXT GENERATION METRICS")
    print('-' * 80)
    print(f"BLEU:    {text_metrics['bleu']:.4f}")
    print(f"ROUGE-1: {text_metrics['rouge_1']:.4f}")
    print(f"ROUGE-L: {text_metrics['rouge_l']:.4f}")
    print(f"METEOR:  {text_metrics['meteor']:.4f}")
    
    return {
        'mcq_accuracy': mcq_acc,
        'bleu': text_metrics['bleu'],
        'rouge_1': text_metrics['rouge_1'],
        'rouge_l': text_metrics['rouge_l'],
        'meteor': text_metrics['meteor'],
    }


def plot_confusion_matrix(cm, model_name, save_path):
    """Plot and save confusion matrix.
    
    Args:
        cm: Confusion matrix
        model_name: Name of the model
        save_path: Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Incorrect', 'Correct'],
        yticklabels=['Incorrect', 'Correct'],
        cbar_kws={'label': 'Count'},
    )
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


def compare_with_supervised_models(kmeans_metrics):
    """Compare K-Means with supervised models.
    
    Args:
        kmeans_metrics: Dictionary of K-Means metrics
    
    Returns:
        DataFrame: Comparison table
    """
    print(f"\n{'=' * 80}")
    print("COMPARISON WITH SUPERVISED MODELS")
    print('=' * 80)
    
    # Load supervised models and evaluate
    models = {}
    try:
        models['LR'] = joblib.load(os.path.join(MODELS_DIR, "lr_model.pkl"))
        print("Loaded Logistic Regression model")
    except:
        print("Warning: Could not load LR model")
    
    try:
        models['SVM'] = joblib.load(os.path.join(MODELS_DIR, "svm_model.pkl"))
        print("Loaded SVM model")
    except:
        print("Warning: Could not load SVM model")
    
    try:
        models['RF'] = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))
        print("Loaded Random Forest model")
    except:
        print("Warning: Could not load RF model")
    
    # Load test data
    X_test = load_npz(os.path.join(PROCESSED_DIR, "X_test.npz"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
    
    # Load feature-only test data for RF (last 5 features)
    X_test_features = X_test[:, -5:].toarray()  # RF uses only lexical features
    
    # Evaluate each model
    results = []
    
    for model_name, model in models.items():
        # RF uses only 5 lexical features, others use full feature matrix
        if model_name == 'RF':
            y_pred = model.predict(X_test_features)
        else:
            y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        results.append({
            'Model': model_name,
            'Type': 'Supervised',
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
        })
    
    # Add K-Means results
    results.append({
        'Model': 'K-Means',
        'Type': 'Unsupervised',
        'Accuracy': kmeans_metrics['accuracy'],
        'Precision': kmeans_metrics['precision'],
        'Recall': kmeans_metrics['recall'],
        'F1-Score': kmeans_metrics['f1_score'],
    })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values('Accuracy', ascending=False)
    
    print(f"\n{df.to_string(index=False)}")
    
    return df


def main():
    """Main evaluation pipeline."""
    print("=" * 80)
    print("K-MEANS CLUSTERING EVALUATION FOR ANSWER VERIFICATION")
    print("=" * 80)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Step 1: Train K-Means model
    print("\n" + "=" * 80)
    print("STEP 1: TRAINING K-MEANS MODEL")
    print("=" * 80)
    
    model = train_kmeans_model(
        processed_dir=PROCESSED_DIR,
        models_dir=MODELS_DIR,
        n_clusters=2,
        random_state=42,
    )
    
    # Step 2: Evaluate on test set (binary classification)
    print("\n" + "=" * 80)
    print("STEP 2: BINARY CLASSIFICATION EVALUATION")
    print("=" * 80)
    
    test_metrics = evaluate_kmeans_model(model, processed_dir=PROCESSED_DIR, split="test")
    
    # Step 3: Plot confusion matrix
    print("\n" + "=" * 80)
    print("STEP 3: GENERATING CONFUSION MATRIX")
    print("=" * 80)
    
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix_kmeans.png")
    plot_confusion_matrix(test_metrics['confusion_matrix'], "K-Means Clustering", cm_path)
    
    # Step 4: Evaluate on MCQ format
    print("\n" + "=" * 80)
    print("STEP 4: MCQ FORMAT EVALUATION")
    print("=" * 80)
    
    test_csv_path = os.path.join(BASE_DIR, "data", "raw", "test.csv")
    mcq_metrics = evaluate_on_mcq_format(model, test_csv_path, model_name="K-Means")
    
    # Step 5: Compare with supervised models
    print("\n" + "=" * 80)
    print("STEP 5: COMPARISON WITH SUPERVISED MODELS")
    print("=" * 80)
    
    comparison_df = compare_with_supervised_models(test_metrics)
    
    # Save comparison table
    comparison_path = os.path.join(RESULTS_DIR, "model_comparison_with_kmeans.csv")
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nComparison table saved to: {comparison_path}")
    
    # Step 6: Summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    
    print("\nK-Means Performance:")
    print(f"  Binary Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Binary Precision: {test_metrics['precision']:.4f}")
    print(f"  Binary Recall:    {test_metrics['recall']:.4f}")
    print(f"  Binary F1-Score:  {test_metrics['f1_score']:.4f}")
    print(f"\n  MCQ Accuracy:     {mcq_metrics['mcq_accuracy']:.4f}")
    print(f"  BLEU:             {mcq_metrics['bleu']:.4f}")
    print(f"  ROUGE-1:          {mcq_metrics['rouge_1']:.4f}")
    print(f"  ROUGE-L:          {mcq_metrics['rouge_l']:.4f}")
    print(f"  METEOR:           {mcq_metrics['meteor']:.4f}")
    
    print("\nCluster Statistics:")
    stats = model.get_cluster_stats()
    for cluster_id, size in stats['cluster_sizes'].items():
        correctness = stats['cluster_correctness'][cluster_id]
        marker = " ← CORRECT CLUSTER" if cluster_id == stats['correct_cluster'] else ""
        print(f"  Cluster {cluster_id}: {size:,} samples, {correctness:.2%} correct{marker}")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {RESULTS_DIR}/")
    print("  - confusion_matrix_kmeans.png")
    print("  - model_comparison_with_kmeans.csv")
    print("  - kmeans_model.pkl (in models/model_a/traditional/)")


if __name__ == "__main__":
    main()
