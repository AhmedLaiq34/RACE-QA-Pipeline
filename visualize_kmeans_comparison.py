"""Generate visualizations comparing K-Means with supervised models."""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

def plot_model_comparison():
    """Create comprehensive comparison visualization."""
    
    # Load comparison data
    df = pd.read_csv(os.path.join(RESULTS_DIR, "all_models_comparison.csv"))
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # 1. MCQ Accuracy Comparison
    ax1 = plt.subplot(2, 3, 1)
    colors = ['#2ecc71' if t == 'Supervised' else '#e74c3c' for t in df['type']]
    bars = ax1.barh(df['model'], df['mcq_accuracy'], color=colors, alpha=0.8)
    ax1.axvline(x=0.25, color='gray', linestyle='--', linewidth=1, label='Random Baseline (25%)')
    ax1.set_xlabel('MCQ Accuracy', fontweight='bold')
    ax1.set_title('MCQ Accuracy Comparison', fontweight='bold', fontsize=12)
    ax1.legend()
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, df['mcq_accuracy'])):
        ax1.text(val + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{val:.2%}', va='center', fontsize=9)
    
    # 2. BLEU Score Comparison
    ax2 = plt.subplot(2, 3, 2)
    bars = ax2.barh(df['model'], df['bleu'], color=colors, alpha=0.8)
    ax2.set_xlabel('BLEU Score', fontweight='bold')
    ax2.set_title('BLEU Score Comparison', fontweight='bold', fontsize=12)
    
    for i, (bar, val) in enumerate(zip(bars, df['bleu'])):
        ax2.text(val + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{val:.4f}', va='center', fontsize=9)
    
    # 3. ROUGE-1 Score Comparison
    ax3 = plt.subplot(2, 3, 3)
    bars = ax3.barh(df['model'], df['rouge_1'], color=colors, alpha=0.8)
    ax3.set_xlabel('ROUGE-1 Score', fontweight='bold')
    ax3.set_title('ROUGE-1 Score Comparison', fontweight='bold', fontsize=12)
    
    for i, (bar, val) in enumerate(zip(bars, df['rouge_1'])):
        ax3.text(val + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{val:.4f}', va='center', fontsize=9)
    
    # 4. METEOR Score Comparison
    ax4 = plt.subplot(2, 3, 4)
    bars = ax4.barh(df['model'], df['meteor'], color=colors, alpha=0.8)
    ax4.set_xlabel('METEOR Score', fontweight='bold')
    ax4.set_title('METEOR Score Comparison', fontweight='bold', fontsize=12)
    
    for i, (bar, val) in enumerate(zip(bars, df['meteor'])):
        ax4.text(val + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{val:.4f}', va='center', fontsize=9)
    
    # 5. All Metrics Heatmap
    ax5 = plt.subplot(2, 3, 5)
    metrics_df = df[['model', 'mcq_accuracy', 'bleu', 'rouge_1', 'rouge_l', 'meteor']].set_index('model')
    sns.heatmap(metrics_df.T, annot=True, fmt='.4f', cmap='YlGnBu', 
                cbar_kws={'label': 'Score'}, ax=ax5)
    ax5.set_title('All Metrics Heatmap', fontweight='bold', fontsize=12)
    ax5.set_xlabel('')
    ax5.set_ylabel('Metric', fontweight='bold')
    
    # 6. Supervised vs Unsupervised Gap
    ax6 = plt.subplot(2, 3, 6)
    
    # Calculate gaps
    best_supervised = df[df['type'] == 'Supervised']['mcq_accuracy'].max()
    kmeans_acc = df[df['model'] == 'K-Means']['mcq_accuracy'].values[0]
    random_baseline = 0.25
    
    categories = ['Random\nBaseline', 'K-Means\n(Unsupervised)', 'Best Supervised\n(SVM)']
    values = [random_baseline, kmeans_acc, best_supervised]
    colors_gap = ['#95a5a6', '#e74c3c', '#2ecc71']
    
    bars = ax6.bar(categories, values, color=colors_gap, alpha=0.8)
    ax6.set_ylabel('MCQ Accuracy', fontweight='bold')
    ax6.set_title('Supervised vs Unsupervised Gap', fontweight='bold', fontsize=12)
    ax6.set_ylim(0, 0.4)
    
    # Add value labels and gaps
    for bar, val in zip(bars, values):
        ax6.text(bar.get_x() + bar.get_width()/2, val + 0.01, 
                f'{val:.2%}', ha='center', fontweight='bold', fontsize=10)
    
    # Add gap annotations
    ax6.annotate('', xy=(1, kmeans_acc), xytext=(1, random_baseline),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax6.text(1.15, (kmeans_acc + random_baseline)/2, 
            f'+{(kmeans_acc - random_baseline)*100:.2f}pp', 
            fontsize=9, fontweight='bold')
    
    ax6.annotate('', xy=(2, best_supervised), xytext=(2, kmeans_acc),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax6.text(2.15, (best_supervised + kmeans_acc)/2, 
            f'+{(best_supervised - kmeans_acc)*100:.2f}pp', 
            fontsize=9, fontweight='bold')
    
    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', alpha=0.8, label='Supervised'),
        Patch(facecolor='#e74c3c', alpha=0.8, label='Unsupervised')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, 
              bbox_to_anchor=(0.5, 0.98), fontsize=11, frameon=True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    output_path = os.path.join(RESULTS_DIR, "kmeans_comprehensive_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_radar_chart():
    """Create radar chart comparing all models."""
    
    df = pd.read_csv(os.path.join(RESULTS_DIR, "all_models_comparison.csv"))
    
    # Prepare data
    categories = ['MCQ Accuracy', 'BLEU', 'ROUGE-1', 'ROUGE-L', 'METEOR']
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Number of variables
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    # Plot each model
    colors_map = {
        'SVM': '#2ecc71',
        'LR': '#3498db',
        'RF': '#9b59b6',
        'Ensemble': '#f39c12',
        'K-Means': '#e74c3c'
    }
    
    for _, row in df.iterrows():
        values = [row['mcq_accuracy'], row['bleu'], row['rouge_1'], 
                 row['rouge_l'], row['meteor']]
        values += values[:1]
        
        color = colors_map.get(row['model'], '#95a5a6')
        linewidth = 3 if row['model'] == 'K-Means' else 2
        linestyle = '--' if row['model'] == 'K-Means' else '-'
        
        ax.plot(angles, values, 'o-', linewidth=linewidth, 
               label=row['model'], color=color, linestyle=linestyle)
        ax.fill(angles, values, alpha=0.15, color=color)
    
    # Fix axis
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 0.5)
    ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_yticklabels(['0.1', '0.2', '0.3', '0.4', '0.5'], fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    
    plt.title('Model Performance Radar Chart\n(K-Means highlighted with dashed line)', 
             fontsize=14, fontweight='bold', pad=20)
    
    # Save
    output_path = os.path.join(RESULTS_DIR, "kmeans_radar_chart.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """Generate all visualizations."""
    print("=" * 80)
    print("GENERATING K-MEANS COMPARISON VISUALIZATIONS")
    print("=" * 80)
    
    print("\n1. Creating comprehensive comparison plot...")
    plot_model_comparison()
    
    print("\n2. Creating radar chart...")
    plot_radar_chart()
    
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"\nSaved to: {RESULTS_DIR}/")
    print("  - kmeans_comprehensive_comparison.png")
    print("  - kmeans_radar_chart.png")


if __name__ == "__main__":
    main()
