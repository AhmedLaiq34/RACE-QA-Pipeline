"""Compare all models (LR, SVM, RF, K-Means, Ensemble) on MCQ accuracy and text metrics.

This script provides a comprehensive comparison of all models including the
unsupervised K-Means clustering approach.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz

# Add pipeline to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pipeline"))

from pipeline.preprocessing import clean_text, expand_df, prepare_text_columns

# Text generation metrics
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    from nltk.translate.meteor_score import meteor_score
    import nltk
    
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
    METRICS_AVAILABLE = False


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models", "model_a", "traditional")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def calculate_text_metrics(predicted_answers, reference_answers):
    """Calculate BLEU, ROUGE, and METEOR scores."""
    if not METRICS_AVAILABLE:
        return {'bleu': 0.0, 'rouge_1': 0.0, 'rouge_l': 0.0, 'meteor': 0.0}
    
    bleu_scores = []
    rouge_1_scores = []
    rouge_l_scores = []
    meteor_scores = []
    
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    smoothing = SmoothingFunction().method1
    
    for pred, ref in zip(predicted_answers, reference_answers):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        
        if len(pred_tokens) > 0 and len(ref_tokens) > 0:
            bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing)
            bleu_scores.append(bleu)
        
        rouge_scores = rouge_scorer_obj.score(ref, pred)
        rouge_1_scores.append(rouge_scores['rouge1'].fmeasure)
        rouge_l_scores.append(rouge_scores['rougeL'].fmeasure)
        
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


def evaluate_model_mcq(model, X_test, test_df, model_name, use_features_only=False):
    """Evaluate a model on MCQ format.
    
    Args:
        model: Trained model
        X_test: Full feature matrix
        test_df: Test DataFrame with MCQ format
        model_name: Name of the model
        use_features_only: If True, use only last 5 features (for RF)
    
    Returns:
        dict: Evaluation metrics
    """
    print(f"\nEvaluating {model_name}...")
    
    # Prepare features
    if use_features_only:
        X = X_test[:, -5:].toarray()
    else:
        X = X_test
    
    # Predict on binary format
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X)[:, 1]
    else:
        y_proba = model.predict(X)
    
    # Calculate MCQ accuracy
    n_questions = len(test_df)
    correct = 0
    predicted_answers = []
    reference_answers = []
    
    for i in range(n_questions):
        start_idx = i * 4
        end_idx = start_idx + 4
        
        # Get predictions for 4 options
        option_probs = y_proba[start_idx:end_idx]
        predicted_option_idx = np.argmax(option_probs)
        
        # Get answer texts
        row = test_df.iloc[i]
        options = ['A', 'B', 'C', 'D']
        predicted_answer = row[options[predicted_option_idx]]
        reference_answer = row[row['answer']]
        
        predicted_answers.append(str(predicted_answer))
        reference_answers.append(str(reference_answer))
        
        # Check if correct
        if options[predicted_option_idx] == row['answer']:
            correct += 1
    
    mcq_accuracy = correct / n_questions
    
    # Calculate text metrics
    text_metrics = calculate_text_metrics(predicted_answers, reference_answers)
    
    return {
        'model': model_name,
        'mcq_accuracy': mcq_accuracy,
        'bleu': text_metrics['bleu'],
        'rouge_1': text_metrics['rouge_1'],
        'rouge_l': text_metrics['rouge_l'],
        'meteor': text_metrics['meteor'],
    }


def main():
    """Main comparison pipeline."""
    print("=" * 80)
    print("COMPREHENSIVE MODEL COMPARISON (INCLUDING K-MEANS)")
    print("=" * 80)
    
    # Load test data
    print("\nLoading test data...")
    test_csv_path = os.path.join(BASE_DIR, "data", "raw", "test.csv")
    test_df = pd.read_csv(test_csv_path)
    
    X_test = load_npz(os.path.join(PROCESSED_DIR, "X_test.npz"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
    
    print(f"Loaded {len(test_df)} questions")
    print(f"Binary samples: {X_test.shape[0]}")
    print(f"Features: {X_test.shape[1]}")
    
    # Load all models
    print("\n" + "=" * 80)
    print("LOADING MODELS")
    print("=" * 80)
    
    models = {}
    
    try:
        models['LR'] = joblib.load(os.path.join(MODELS_DIR, "lr_model.pkl"))
        print("✓ Loaded Logistic Regression")
    except Exception as e:
        print(f"✗ Could not load LR: {e}")
    
    try:
        models['SVM'] = joblib.load(os.path.join(MODELS_DIR, "svm_model.pkl"))
        print("✓ Loaded SVM")
    except Exception as e:
        print(f"✗ Could not load SVM: {e}")
    
    try:
        models['RF'] = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))
        print("✓ Loaded Random Forest")
    except Exception as e:
        print(f"✗ Could not load RF: {e}")
    
    try:
        models['K-Means'] = joblib.load(os.path.join(MODELS_DIR, "kmeans_model.pkl"))
        print("✓ Loaded K-Means")
    except Exception as e:
        print(f"✗ Could not load K-Means: {e}")
    
    # Evaluate all models
    print("\n" + "=" * 80)
    print("EVALUATING ALL MODELS")
    print("=" * 80)
    
    results = []
    
    for model_name, model in models.items():
        use_features_only = (model_name == 'RF')
        metrics = evaluate_model_mcq(model, X_test, test_df, model_name, use_features_only)
        results.append(metrics)
    
    # Create ensemble (SVM × 0.6 + LR × 0.4)
    if 'SVM' in models and 'LR' in models:
        print("\nEvaluating Ensemble (SVM×0.6 + LR×0.4)...")
        
        svm_proba = models['SVM'].predict_proba(X_test)[:, 1]
        lr_proba = models['LR'].predict_proba(X_test)[:, 1]
        ensemble_proba = 0.6 * svm_proba + 0.4 * lr_proba
        
        n_questions = len(test_df)
        correct = 0
        predicted_answers = []
        reference_answers = []
        
        for i in range(n_questions):
            start_idx = i * 4
            end_idx = start_idx + 4
            
            option_probs = ensemble_proba[start_idx:end_idx]
            predicted_option_idx = np.argmax(option_probs)
            
            row = test_df.iloc[i]
            options = ['A', 'B', 'C', 'D']
            predicted_answer = row[options[predicted_option_idx]]
            reference_answer = row[row['answer']]
            
            predicted_answers.append(str(predicted_answer))
            reference_answers.append(str(reference_answer))
            
            if options[predicted_option_idx] == row['answer']:
                correct += 1
        
        mcq_accuracy = correct / n_questions
        text_metrics = calculate_text_metrics(predicted_answers, reference_answers)
        
        results.append({
            'model': 'Ensemble',
            'mcq_accuracy': mcq_accuracy,
            'bleu': text_metrics['bleu'],
            'rouge_1': text_metrics['rouge_1'],
            'rouge_l': text_metrics['rouge_l'],
            'meteor': text_metrics['meteor'],
        })
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values('mcq_accuracy', ascending=False)
    
    # Display results
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    print("\n" + df.to_string(index=False))
    
    # Add model type column
    df['type'] = df['model'].apply(lambda x: 'Unsupervised' if x == 'K-Means' else 'Supervised')
    
    # Reorder columns
    df = df[['model', 'type', 'mcq_accuracy', 'bleu', 'rouge_1', 'rouge_l', 'meteor']]
    
    # Save results
    output_path = os.path.join(RESULTS_DIR, "all_models_comparison.csv")
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # Print key insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    
    best_supervised = df[df['type'] == 'Supervised'].iloc[0]
    kmeans_row = df[df['model'] == 'K-Means'].iloc[0]
    
    print(f"\nBest Supervised Model: {best_supervised['model']}")
    print(f"  MCQ Accuracy: {best_supervised['mcq_accuracy']:.4f} ({best_supervised['mcq_accuracy']*100:.2f}%)")
    
    print(f"\nK-Means (Unsupervised):")
    print(f"  MCQ Accuracy: {kmeans_row['mcq_accuracy']:.4f} ({kmeans_row['mcq_accuracy']*100:.2f}%)")
    
    print(f"\nRandom Baseline: 0.2500 (25.00%)")
    
    gap = best_supervised['mcq_accuracy'] - kmeans_row['mcq_accuracy']
    print(f"\nSupervised vs Unsupervised Gap: {gap:.4f} ({gap*100:.2f} percentage points)")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
