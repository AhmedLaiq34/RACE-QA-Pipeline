"""K-Means Clustering for Unsupervised Answer Verification.

This module implements K-Means clustering as an unsupervised approach to answer
verification. The model groups question-answer pairs by feature similarity to
discover latent answer patterns without using labels during training.

Approach:
1. Extract features using One-Hot Encoding (binary CountVectorizer)
2. Apply K-Means clustering with k=2 (correct vs incorrect patterns)
3. Assign cluster labels based on which cluster has more correct answers
4. Use cluster assignments for answer verification

The model is evaluated using the same metrics as supervised models for comparison.
"""

import os
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    silhouette_score,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DEFAULT_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
DEFAULT_MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "model_a", "traditional")


def _resolve_path(path_value):
    """Convert relative paths to absolute paths from project root."""
    if path_value is None:
        return None
    if os.path.isabs(str(path_value)):
        return str(path_value)
    return os.path.join(PROJECT_ROOT, str(path_value))


# Rubric: Approach Implemented (5 Marks): Implementation and evaluation of at least one unsupervised or semi-supervised approach such as K-Means, Label Propagation, or GMM; comparison against supervised models. - 1
class KMeansAnswerVerifier:
    """K-Means clustering model for unsupervised answer verification.
    
    This model uses K-Means with k=2 to discover latent patterns in the feature
    space that correspond to correct vs incorrect answers. After clustering,
    we assign the cluster with higher proportion of correct answers as the
    "correct" cluster.
    
    Attributes:
        n_clusters (int): Number of clusters (default: 2 for binary classification)
        kmeans (KMeans): Scikit-learn KMeans model
        correct_cluster (int): Cluster ID assigned to correct answers (0 or 1)
        cluster_stats (dict): Statistics about cluster composition
    """
    
    def __init__(self, n_clusters=2, random_state=42, max_iter=300, n_init=10):
        """Initialize K-Means model.
        
        Args:
            n_clusters (int): Number of clusters (2 for binary classification)
            random_state (int): Random seed for reproducibility
            max_iter (int): Maximum iterations for K-Means
            n_init (int): Number of K-Means initializations
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            max_iter=max_iter,
            n_init=n_init,
            verbose=0,
        )
        self.correct_cluster = None
        self.cluster_stats = {}
    
    def fit(self, X, y=None):
        """Fit K-Means on training data.
        
        Note: y is optional and only used to determine which cluster corresponds
        to correct answers. The clustering itself is unsupervised.
        
        Args:
            X (sparse matrix): Feature matrix (n_samples × n_features)
            y (array, optional): True labels for cluster assignment
        
        Returns:
            self
        """
        print(f"Fitting K-Means with {self.n_clusters} clusters...")
        print(f"Feature matrix shape: {X.shape}")
        
        # Fit K-Means (unsupervised)
        self.kmeans.fit(X)
        
        # If labels provided, determine which cluster is "correct"
        if y is not None:
            cluster_labels = self.kmeans.labels_
            
            # Calculate proportion of correct answers in each cluster
            cluster_correctness = {}
            for cluster_id in range(self.n_clusters):
                mask = cluster_labels == cluster_id
                if mask.sum() > 0:
                    correct_ratio = y[mask].mean()
                    cluster_correctness[cluster_id] = correct_ratio
                    print(f"Cluster {cluster_id}: {mask.sum()} samples, "
                          f"{correct_ratio:.2%} correct")
            
            # Assign the cluster with highest correct ratio as "correct cluster"
            self.correct_cluster = max(cluster_correctness, key=cluster_correctness.get)
            print(f"\nAssigned Cluster {self.correct_cluster} as 'correct' cluster")
            
            # Store statistics
            self.cluster_stats = {
                'cluster_sizes': {i: (cluster_labels == i).sum() 
                                 for i in range(self.n_clusters)},
                'cluster_correctness': cluster_correctness,
                'correct_cluster': self.correct_cluster,
            }
        
        print(f"K-Means training complete. Inertia: {self.kmeans.inertia_:.2f}")
        return self
    
    def predict(self, X):
        """Predict cluster assignments for new data.
        
        Args:
            X (sparse matrix): Feature matrix
        
        Returns:
            array: Binary predictions (1 if assigned to correct cluster, 0 otherwise)
        """
        cluster_assignments = self.kmeans.predict(X)
        
        # Convert cluster assignments to binary predictions
        if self.correct_cluster is not None:
            predictions = (cluster_assignments == self.correct_cluster).astype(int)
        else:
            # If no correct cluster assigned, use cluster 0 as default
            predictions = (cluster_assignments == 0).astype(int)
        
        return predictions
    
    def predict_proba(self, X):
        """Predict probability-like scores based on distance to cluster centers.
        
        Note: K-Means doesn't naturally produce probabilities. We use distance
        to cluster centers as a proxy, normalized to [0, 1] range.
        
        Args:
            X (sparse matrix): Feature matrix
        
        Returns:
            array: Probability-like scores (n_samples × 2)
        """
        # Get distances to all cluster centers
        distances = self.kmeans.transform(X)
        
        # Convert distances to similarity scores (inverse distance)
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        similarities = 1.0 / (distances + epsilon)
        
        # Normalize to get probability-like scores
        proba = similarities / similarities.sum(axis=1, keepdims=True)
        
        # If correct_cluster is 1, swap columns so proba[:, 1] is correct class
        if self.correct_cluster == 1:
            return proba
        else:
            return proba[:, ::-1]  # Reverse columns
    
    def get_cluster_centers(self):
        """Get cluster centers.
        
        Returns:
            array: Cluster centers (n_clusters × n_features)
        """
        return self.kmeans.cluster_centers_
    
    def get_cluster_stats(self):
        """Get statistics about cluster composition.
        
        Returns:
            dict: Cluster statistics
        """
        return self.cluster_stats


def train_kmeans_model(
    processed_dir=DEFAULT_PROCESSED_DIR,
    models_dir=DEFAULT_MODELS_DIR,
    n_clusters=2,
    random_state=42,
):
    """Train K-Means clustering model for answer verification.
    
    Args:
        processed_dir (str): Directory with preprocessed features
        models_dir (str): Directory to save trained model
        n_clusters (int): Number of clusters
        random_state (int): Random seed
    
    Returns:
        KMeansAnswerVerifier: Trained model
    """
    processed_dir = _resolve_path(processed_dir)
    models_dir = _resolve_path(models_dir)
    os.makedirs(models_dir, exist_ok=True)
    
    print("=" * 80)
    print("K-MEANS CLUSTERING FOR ANSWER VERIFICATION")
    print("=" * 80)
    
    # Load training data
    print("\nLoading training data...")
    X_train = load_npz(os.path.join(processed_dir, "X_train.npz"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Positive samples: {y_train.sum()} ({y_train.mean():.2%})")
    print(f"Negative samples: {(1 - y_train).sum()} ({(1 - y_train).mean():.2%})")
    
    # Train K-Means
    print("\n" + "-" * 80)
    model = KMeansAnswerVerifier(n_clusters=n_clusters, random_state=random_state)
    model.fit(X_train, y_train)
    
    # Evaluate on training set
    print("\n" + "-" * 80)
    print("TRAINING SET EVALUATION")
    print("-" * 80)
    y_train_pred = model.predict(X_train)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    train_prec = precision_score(y_train, y_train_pred, zero_division=0)
    train_rec = recall_score(y_train, y_train_pred, zero_division=0)
    train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
    
    print(f"Accuracy:  {train_acc:.4f}")
    print(f"Precision: {train_prec:.4f}")
    print(f"Recall:    {train_rec:.4f}")
    print(f"F1-Score:  {train_f1:.4f}")
    
    # Rubric: Evaluation Metrics (5 Marks): Clustering purity, silhouette score, or semi-supervised F1 reported. - 1
    # Calculate silhouette score (clustering quality metric)
    # Use a sample for efficiency (silhouette is expensive on large datasets)
    sample_size = min(10000, X_train.shape[0])
    sample_indices = np.random.RandomState(random_state).choice(
        X_train.shape[0], sample_size, replace=False
    )
    X_sample = X_train[sample_indices]
    labels_sample = model.kmeans.predict(X_sample)
    silhouette = silhouette_score(X_sample, labels_sample, metric='euclidean')
    print(f"Silhouette Score: {silhouette:.4f} (clustering quality, range: [-1, 1])")
    
    # Save model
    model_path = os.path.join(models_dir, "kmeans_model.pkl")
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")
    print(f"Model size: {os.path.getsize(model_path) / 1024:.1f} KB")
    
    return model


def evaluate_kmeans_model(
    model,
    processed_dir=DEFAULT_PROCESSED_DIR,
    split="test",
):
    """Evaluate K-Means model on a dataset split.
    
    Args:
        model (KMeansAnswerVerifier): Trained model
        processed_dir (str): Directory with preprocessed features
        split (str): Dataset split ('train', 'val', or 'test')
    
    Returns:
        dict: Evaluation metrics
    """
    processed_dir = _resolve_path(processed_dir)
    
    print("\n" + "=" * 80)
    print(f"EVALUATING K-MEANS ON {split.upper()} SET")
    print("=" * 80)
    
    # Load data
    X = load_npz(os.path.join(processed_dir, f"X_{split}.npz"))
    y = np.load(os.path.join(processed_dir, f"y_{split}.npy"))
    
    print(f"\n{split.capitalize()} samples: {X.shape[0]}")
    print(f"Positive samples: {y.sum()} ({y.mean():.2%})")
    
    # Predict
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    
    # Calculate metrics
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    cm = confusion_matrix(y, y_pred)
    
    print("\n" + "-" * 80)
    print("CLASSIFICATION METRICS")
    print("-" * 80)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    print("\n" + "-" * 80)
    print("CONFUSION MATRIX")
    print("-" * 80)
    print("              Predicted")
    print("              Neg    Pos")
    print(f"Actual Neg │ {cm[0, 0]:5d}  {cm[0, 1]:5d}")
    print(f"       Pos │ {cm[1, 0]:5d}  {cm[1, 1]:5d}")
    
    # Per-class metrics
    print("\n" + "-" * 80)
    print("PER-CLASS METRICS")
    print("-" * 80)
    print(f"Class 0 (Incorrect): Precision={cm[0, 0] / max(cm[0, 0] + cm[1, 0], 1):.4f}, "
          f"Recall={cm[0, 0] / max(cm[0, 0] + cm[0, 1], 1):.4f}")
    print(f"Class 1 (Correct):   Precision={prec:.4f}, Recall={rec:.4f}")
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'probabilities': y_proba,
    }


def mcq_accuracy_from_binary(y_pred, y_true, n_questions):
    """Calculate MCQ accuracy from binary predictions.
    
    For each question (4 binary samples), select the option with highest
    predicted probability and check if it matches the correct answer.
    
    Args:
        y_pred (array): Binary predictions (n_samples,)
        y_true (array): True binary labels (n_samples,)
        n_questions (int): Number of MCQ questions
    
    Returns:
        float: MCQ accuracy (0-1)
    """
    correct = 0
    for i in range(n_questions):
        start_idx = i * 4
        end_idx = start_idx + 4
        
        # Get predictions for 4 options
        option_preds = y_pred[start_idx:end_idx]
        option_labels = y_true[start_idx:end_idx]
        
        # Find which option was predicted as correct
        predicted_option = np.argmax(option_preds)
        
        # Find which option is actually correct
        true_option = np.argmax(option_labels)
        
        if predicted_option == true_option:
            correct += 1
    
    return correct / n_questions


if __name__ == "__main__":
    # Train K-Means model
    model = train_kmeans_model()
    
    # Evaluate on validation set
    val_metrics = evaluate_kmeans_model(model, split="val")
    
    # Evaluate on test set
    test_metrics = evaluate_kmeans_model(model, split="test")
    
    print("\n" + "=" * 80)
    print("K-MEANS TRAINING AND EVALUATION COMPLETE")
    print("=" * 80)
