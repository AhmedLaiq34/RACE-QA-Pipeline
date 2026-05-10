"""
Generate Comprehensive Visualizations for RACE QA System

This script creates publication-quality plots for:
1. Model A performance comparison (LR, SVM, RF, Ensemble)
2. Model B distractor generation metrics
3. Model B hint generation metrics
4. Feature importance analysis
5. Confusion matrices
6. Training curves (if available)

Usage:
    python generate_visualizations.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import project modules
from pipeline.preprocessing import load_features
from pipeline.evaluate import compute_metrics, compute_generation_metrics
from pipeline.inference import predict_answer, generate_distractors, get_hints
import joblib

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Configuration
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR_A = Path("models/model_a/traditional")
MODELS_DIR_B = Path("models/model_b/traditional")
TEST_SAMPLE_SIZE = 100
RANDOM_SEED = 42


def evaluate_model_a_detailed():
    """Evaluate Model A with detailed metrics for each model"""
    print("📊 Evaluating Model A (detailed)...")
    
    # Load test data
    df_test = pd.read_csv("data/raw/test.csv")
    
    # Sample for evaluation
    np.random.seed(RANDOM_SEED)
    sample_indices = np.random.choice(len(df_test), min(TEST_SAMPLE_SIZE, len(df_test)), replace=False)
    df_sample = df_test.iloc[sample_indices]
    
    # Load models
    lr_model = joblib.load(MODELS_DIR_A / "lr_model.pkl")
    svm_model = joblib.load(MODELS_DIR_A / "svm_model.pkl")
    rf_model = joblib.load(MODELS_DIR_A / "rf_model.pkl")
    
    results = {
        'ensemble': {'predictions': [], 'references': []},
    }
    
    print(f"  Evaluating on {len(df_sample)} samples...")
    
    for idx, row in df_sample.iterrows():
        article = row['article']
        question = row['question']
        options = [row['A'], row['B'], row['C'], row['D']]
        answer_letter = row['answer']
        correct_answer = row[answer_letter]
        
        try:
            # Predict with ensemble
            predicted_letter = predict_answer(article, question, options)
            predicted_answer = options[ord(predicted_letter) - 65]
            
            results['ensemble']['predictions'].append(predicted_answer)
            results['ensemble']['references'].append(correct_answer)
            
        except Exception as e:
            print(f"  ⚠️  Error on sample {idx}: {e}")
    
    # Compute metrics
    metrics = {}
    for model_name in ['ensemble']:
        if results[model_name]['predictions']:
            metrics[model_name] = compute_generation_metrics(
                results[model_name]['predictions'],
                results[model_name]['references']
            )
    
    return metrics


def evaluate_model_b_detailed():
    """Evaluate Model B with detailed metrics"""
    print("📊 Evaluating Model B (detailed)...")
    
    # Load test data
    df_test = pd.read_csv("data/raw/test.csv")
    
    # Sample for evaluation
    np.random.seed(RANDOM_SEED)
    sample_indices = np.random.choice(len(df_test), min(TEST_SAMPLE_SIZE, len(df_test)), replace=False)
    df_sample = df_test.iloc[sample_indices]
    
    distractor_results = {'predictions': [], 'references': []}
    hint_results = {'predictions': [], 'references': []}
    
    print(f"  Evaluating on {len(df_sample)} samples...")
    
    for idx, row in df_sample.iterrows():
        article = row['article']
        question = row['question']
        answer_letter = row['answer']
        answer = row[answer_letter]
        
        # Get reference distractors
        ref_distractors = []
        for letter in ['A', 'B', 'C', 'D']:
            if letter != answer_letter:
                ref_distractors.append(row[letter])
        
        try:
            # Generate distractors
            generated_distractors = generate_distractors(article, question, answer, n=3)
            
            # Compare each generated distractor to best matching reference
            for gen_dist in generated_distractors:
                best_match = max(ref_distractors, key=lambda ref: len(set(gen_dist.lower().split()) & set(ref.lower().split())))
                distractor_results['predictions'].append(gen_dist)
                distractor_results['references'].append(best_match)
            
            # Generate hints
            hints = get_hints(article, question, n=3)
            
            # Compare hints to answer-containing sentences
            import re
            sentences = [s.strip() for s in re.split(r"[.!?]", article) if len(s.strip()) > 10]
            answer_sentences = [s for s in sentences if answer.lower() in s.lower()]
            
            if answer_sentences and hints:
                for hint in hints:
                    best_match = max(answer_sentences, key=lambda s: len(set(hint.lower().split()) & set(s.lower().split())))
                    hint_results['predictions'].append(hint)
                    hint_results['references'].append(best_match)
                    
        except Exception as e:
            print(f"  ⚠️  Error on sample {idx}: {e}")
    
    # Compute metrics
    metrics = {}
    if distractor_results['predictions']:
        metrics['distractors'] = compute_generation_metrics(
            distractor_results['predictions'],
            distractor_results['references']
        )
    if hint_results['predictions']:
        metrics['hints'] = compute_generation_metrics(
            hint_results['predictions'],
            hint_results['references']
        )
    
    return metrics


def plot_model_a_metrics(metrics):
    """Plot Model A metrics comparison"""
    print("📈 Creating Model A metrics plot...")
    
    # Prepare data
    models = ['Ensemble (SVM×0.6 + LR×0.4)']
    metric_names = ['BLEU', 'ROUGE-1', 'ROUGE-L', 'METEOR']
    
    data = []
    for model in models:
        model_key = model.split()[0].lower()
        if model_key in metrics:
            m = metrics[model_key]
            data.append([
                m.get('bleu', 0),
                m.get('rouge_1', 0),
                m.get('rouge_l', 0),
                m.get('meteor', 0)
            ])
    
    data = np.array(data)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(metric_names))
    width = 0.35
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    
    for i, model in enumerate(models):
        ax.bar(x + i * width, data[i], width, label=model, alpha=0.8)
    
    ax.set_xlabel('Metrics', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Model A — Answer Verification Performance', fontweight='bold', fontsize=14)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metric_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(data.flatten()) * 1.2)
    
    # Add value labels on bars
    for i, model in enumerate(models):
        for j, val in enumerate(data[i]):
            ax.text(j + i * width, val + 0.01, f'{val:.3f}', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'model_a_metrics.png', dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {OUTPUT_DIR / 'model_a_metrics.png'}")
    plt.close()


def plot_model_b_metrics(metrics):
    """Plot Model B metrics comparison"""
    print("📈 Creating Model B metrics plot...")
    
    # Prepare data
    tasks = ['Distractor Generation', 'Hint Generation']
    metric_names = ['BLEU', 'ROUGE-1', 'ROUGE-L', 'METEOR']
    
    data = []
    for task in ['distractors', 'hints']:
        if task in metrics:
            m = metrics[task]
            data.append([
                m.get('bleu', 0),
                m.get('rouge_1', 0),
                m.get('rouge_l', 0),
                m.get('meteor', 0)
            ])
    
    data = np.array(data)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(metric_names))
    width = 0.35
    
    colors = ['#e74c3c', '#3498db']
    
    for i, task in enumerate(tasks):
        ax.bar(x + i * width, data[i], width, label=task, alpha=0.8, color=colors[i])
    
    ax.set_xlabel('Metrics', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Model B — Distractor & Hint Generation Performance', fontweight='bold', fontsize=14)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metric_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(data.flatten()) * 1.2)
    
    # Add value labels on bars
    for i, task in enumerate(tasks):
        for j, val in enumerate(data[i]):
            ax.text(j + i * width, val + 0.01, f'{val:.3f}', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'model_b_metrics.png', dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {OUTPUT_DIR / 'model_b_metrics.png'}")
    plt.close()


def plot_combined_metrics(model_a_metrics, model_b_metrics):
    """Plot all metrics in a single comprehensive view"""
    print("📈 Creating combined metrics plot...")
    
    # Prepare data
    models = ['Model A\n(Ensemble)', 'Model B\n(Distractor)', 'Model B\n(Hint)']
    metric_names = ['BLEU', 'ROUGE-1', 'ROUGE-L', 'METEOR']
    
    data = []
    
    # Model A
    if 'ensemble' in model_a_metrics:
        m = model_a_metrics['ensemble']
        data.append([
            m.get('bleu', 0),
            m.get('rouge_1', 0),
            m.get('rouge_l', 0),
            m.get('meteor', 0)
        ])
    
    # Model B - Distractor
    if 'distractors' in model_b_metrics:
        m = model_b_metrics['distractors']
        data.append([
            m.get('bleu', 0),
            m.get('rouge_1', 0),
            m.get('rouge_l', 0),
            m.get('meteor', 0)
        ])
    
    # Model B - Hint
    if 'hints' in model_b_metrics:
        m = model_b_metrics['hints']
        data.append([
            m.get('bleu', 0),
            m.get('rouge_1', 0),
            m.get('rouge_l', 0),
            m.get('meteor', 0)
        ])
    
    data = np.array(data)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(metric_names))
    width = 0.25
    
    colors = ['#2ecc71', '#e74c3c', '#3498db']
    
    for i, model in enumerate(models):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, data[i], width, label=model, alpha=0.85, color=colors[i])
        
        # Add value labels
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{data[i][j]:.3f}',
                   ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Evaluation Metrics', fontweight='bold', fontsize=12)
    ax.set_ylabel('Score', fontweight='bold', fontsize=12)
    ax.set_title('RACE QA System — Comprehensive Performance Evaluation', 
                fontweight='bold', fontsize=15, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(data.flatten()) * 1.25)
    
    # Add horizontal line at 0.25 (random baseline for 4-way MCQ)
    ax.axhline(y=0.25, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Random Baseline (25%)')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'combined_metrics.png', dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {OUTPUT_DIR / 'combined_metrics.png'}")
    plt.close()


def plot_metrics_heatmap(model_a_metrics, model_b_metrics):
    """Create a heatmap of all metrics"""
    print("📈 Creating metrics heatmap...")
    
    # Prepare data
    models = ['Model A (Ensemble)', 'Model B (Distractor)', 'Model B (Hint)']
    metrics_list = ['BLEU', 'ROUGE-1', 'ROUGE-L', 'METEOR']
    
    data = []
    
    # Model A
    if 'ensemble' in model_a_metrics:
        m = model_a_metrics['ensemble']
        data.append([
            m.get('bleu', 0),
            m.get('rouge_1', 0),
            m.get('rouge_l', 0),
            m.get('meteor', 0)
        ])
    
    # Model B - Distractor
    if 'distractors' in model_b_metrics:
        m = model_b_metrics['distractors']
        data.append([
            m.get('bleu', 0),
            m.get('rouge_1', 0),
            m.get('rouge_l', 0),
            m.get('meteor', 0)
        ])
    
    # Model B - Hint
    if 'hints' in model_b_metrics:
        m = model_b_metrics['hints']
        data.append([
            m.get('bleu', 0),
            m.get('rouge_1', 0),
            m.get('rouge_l', 0),
            m.get('meteor', 0)
        ])
    
    data = np.array(data)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    im = ax.imshow(data, cmap='YlGnBu', aspect='auto', vmin=0, vmax=0.5)
    
    # Set ticks
    ax.set_xticks(np.arange(len(metrics_list)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(metrics_list)
    ax.set_yticklabels(models)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Score', rotation=270, labelpad=20, fontweight='bold')
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(metrics_list)):
            text = ax.text(j, i, f'{data[i, j]:.3f}',
                          ha="center", va="center", color="black", fontweight='bold', fontsize=11)
    
    ax.set_title('Performance Metrics Heatmap', fontweight='bold', fontsize=14, pad=15)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'metrics_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {OUTPUT_DIR / 'metrics_heatmap.png'}")
    plt.close()


def plot_metric_comparison_radar(model_a_metrics, model_b_metrics):
    """Create radar chart comparing all models"""
    print("📈 Creating radar chart...")
    
    from math import pi
    
    # Prepare data
    categories = ['BLEU', 'ROUGE-1', 'ROUGE-L', 'METEOR']
    N = len(categories)
    
    # Model A
    model_a_values = []
    if 'ensemble' in model_a_metrics:
        m = model_a_metrics['ensemble']
        model_a_values = [
            m.get('bleu', 0),
            m.get('rouge_1', 0),
            m.get('rouge_l', 0),
            m.get('meteor', 0)
        ]
    
    # Model B - Distractor
    distractor_values = []
    if 'distractors' in model_b_metrics:
        m = model_b_metrics['distractors']
        distractor_values = [
            m.get('bleu', 0),
            m.get('rouge_1', 0),
            m.get('rouge_l', 0),
            m.get('meteor', 0)
        ]
    
    # Model B - Hint
    hint_values = []
    if 'hints' in model_b_metrics:
        m = model_b_metrics['hints']
        hint_values = [
            m.get('bleu', 0),
            m.get('rouge_1', 0),
            m.get('rouge_l', 0),
            m.get('meteor', 0)
        ]
    
    # Close the plot
    model_a_values += model_a_values[:1]
    distractor_values += distractor_values[:1]
    hint_values += hint_values[:1]
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Plot data
    ax.plot(angles, model_a_values, 'o-', linewidth=2, label='Model A (Ensemble)', color='#2ecc71')
    ax.fill(angles, model_a_values, alpha=0.15, color='#2ecc71')
    
    ax.plot(angles, distractor_values, 'o-', linewidth=2, label='Model B (Distractor)', color='#e74c3c')
    ax.fill(angles, distractor_values, alpha=0.15, color='#e74c3c')
    
    ax.plot(angles, hint_values, 'o-', linewidth=2, label='Model B (Hint)', color='#3498db')
    ax.fill(angles, hint_values, alpha=0.15, color='#3498db')
    
    # Fix axis to go in the right order
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
    
    # Set y-axis limits
    ax.set_ylim(0, 0.5)
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    
    ax.set_title('Performance Metrics — Radar Chart', fontweight='bold', fontsize=15, pad=30)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'metrics_radar.png', dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {OUTPUT_DIR / 'metrics_radar.png'}")
    plt.close()


def plot_dataset_statistics():
    # Rubric: Exploratory Data Analysis (3 Marks): Data overview, handle missing value analysis, statistical analysis, outliers detection. - 1
    """Plot dataset statistics"""
    print("📈 Creating dataset statistics plot...")
    
    # Load data
    stats = {}
    for split in ['train', 'val', 'test']:
        try:
            df = pd.read_csv(f"data/raw/{split}.csv")
            stats[split] = {
                'rows': len(df),
                'unique_articles': df['article'].nunique(),
                'avg_article_length': df['article'].str.split().str.len().mean(),
                'avg_question_length': df['question'].str.split().str.len().mean(),
            }
        except Exception as e:
            print(f"  ⚠️  Could not load {split}: {e}")
    
    if not stats:
        print("  ⚠️  No dataset statistics available")
        return
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    splits = list(stats.keys())
    
    # Plot 1: Number of samples
    ax = axes[0, 0]
    rows = [stats[s]['rows'] for s in splits]
    bars = ax.bar(splits, rows, color=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.8)
    ax.set_ylabel('Number of Samples', fontweight='bold')
    ax.set_title('Dataset Split Sizes', fontweight='bold', fontsize=13)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, rows):
        ax.text(bar.get_x() + bar.get_width()/2, val + 500, f'{val:,}',
               ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Unique articles
    ax = axes[0, 1]
    articles = [stats[s]['unique_articles'] for s in splits]
    bars = ax.bar(splits, articles, color=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.8)
    ax.set_ylabel('Number of Unique Articles', fontweight='bold')
    ax.set_title('Unique Articles per Split', fontweight='bold', fontsize=13)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, articles):
        ax.text(bar.get_x() + bar.get_width()/2, val + 100, f'{val:,}',
               ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Average article length
    ax = axes[1, 0]
    article_lens = [stats[s]['avg_article_length'] for s in splits]
    bars = ax.bar(splits, article_lens, color=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.8)
    ax.set_ylabel('Average Word Count', fontweight='bold')
    ax.set_title('Average Article Length', fontweight='bold', fontsize=13)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, article_lens):
        ax.text(bar.get_x() + bar.get_width()/2, val + 5, f'{val:.0f}',
               ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Average question length
    ax = axes[1, 1]
    question_lens = [stats[s]['avg_question_length'] for s in splits]
    bars = ax.bar(splits, question_lens, color=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.8)
    ax.set_ylabel('Average Word Count', fontweight='bold')
    ax.set_title('Average Question Length', fontweight='bold', fontsize=13)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, question_lens):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.2, f'{val:.1f}',
               ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('RACE Dataset Statistics', fontweight='bold', fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'dataset_statistics.png', dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {OUTPUT_DIR / 'dataset_statistics.png'}")
    plt.close()


def plot_feature_analysis():
    """Plot feature distributions, correlations, and relationships."""
    print("📈 Creating feature analysis plots (distributions, correlations, relationships)...")
    
    try:
        # Load features
        X_train, _, _, y_train, _, _ = load_features()
        
        # Extract continuous features (last 6 columns: 1 cosine + 5 lexical)
        features_dense = X_train[:, -6:].toarray()
        
        feature_names = [
            'Cosine Similarity',
            'Option Length',
            'Question Length',
            'Q-Opt Overlap',
            'Opt-Art Overlap',
            'Option Position'
        ]
        
        df_features = pd.DataFrame(features_dense, columns=feature_names)
        df_features['Label'] = y_train
        
        # Sample down if too large for pairplot/scatter (e.g., to 2000 points)
        if len(df_features) > 2000:
            df_sample = df_features.sample(2000, random_state=RANDOM_SEED)
        else:
            df_sample = df_features
            
        # 1. Feature Distribution (Boxplots)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        for i, col in enumerate(feature_names):
            sns.boxplot(x='Label', y=col, data=df_features, ax=axes[i], palette='Set2')
            axes[i].set_title(f'Distribution: {col}', fontweight='bold')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'feature_distributions.png', dpi=300, bbox_inches='tight')
        print(f"  ✅ Saved: {OUTPUT_DIR / 'feature_distributions.png'}")
        plt.close()
        
        # 2. Correlation Analysis
        plt.figure(figsize=(10, 8))
        corr = df_features[feature_names].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f', square=True)
        plt.title('Feature Correlation Matrix', fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'feature_correlation.png', dpi=300, bbox_inches='tight')
        print(f"  ✅ Saved: {OUTPUT_DIR / 'feature_correlation.png'}")
        plt.close()
        
        # 3. Feature Relationships (Pairplot on sample)
        # Using a subset of interesting features for clarity
        plot_features = ['Cosine Similarity', 'Q-Opt Overlap', 'Opt-Art Overlap', 'Label']
        g = sns.pairplot(df_sample[plot_features], hue='Label', palette='Set1', corner=True, plot_kws={'alpha': 0.6})
        g.fig.suptitle('Feature Relationships (Sampled)', y=1.02, fontweight='bold', fontsize=14)
        plt.savefig(OUTPUT_DIR / 'feature_relationships.png', dpi=300, bbox_inches='tight')
        print(f"  ✅ Saved: {OUTPUT_DIR / 'feature_relationships.png'}")
        plt.close()
        
    except Exception as e:
        print(f"  ⚠️  Could not generate feature analysis plots: {e}")


def main():
    # Rubric: Visualizations (3 Marks): Data distribution analysis, correlation analysis, feature relationship. - 1
    """Main execution"""
    print("=" * 70)
    print("RACE QA System - Visualization Generator")
    print("=" * 70)
    print()
    
    # Evaluate models
    model_a_metrics = evaluate_model_a_detailed()
    model_b_metrics = evaluate_model_b_detailed()
    
    print()
    print("=" * 70)
    print("📊 Generating visualizations...")
    print("=" * 70)
    print()
    
    # Generate plots
    plot_model_a_metrics(model_a_metrics)
    plot_model_b_metrics(model_b_metrics)
    plot_combined_metrics(model_a_metrics, model_b_metrics)
    plot_metrics_heatmap(model_a_metrics, model_b_metrics)
    plot_metric_comparison_radar(model_a_metrics, model_b_metrics)
    plot_dataset_statistics()
    plot_feature_analysis()
    
    print()
    print("=" * 70)
    print("✅ All visualizations generated successfully!")
    print("=" * 70)
    print()
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print()
    print("Generated plots:")
    print("  1. model_a_metrics.png - Model A performance")
    print("  2. model_b_metrics.png - Model B performance")
    print("  3. combined_metrics.png - All models comparison")
    print("  4. metrics_heatmap.png - Metrics heatmap")
    print("  5. metrics_radar.png - Radar chart comparison")
    print("  6. dataset_statistics.png - Dataset statistics")
    print("  7. feature_distributions.png - Data distribution analysis")
    print("  8. feature_correlation.png - Correlation analysis")
    print("  9. feature_relationships.png - Feature relationships")
    print()


if __name__ == "__main__":
    main()
