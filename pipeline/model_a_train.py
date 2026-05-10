"""Train Model A answer verification classifiers."""

import os
import time
import warnings

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

try:
    from .evaluate import compute_metrics
    from .preprocessing import DEFAULT_PROCESSED_DIR, PROJECT_ROOT, load_features
except ImportError:
    from evaluate import compute_metrics
    from preprocessing import DEFAULT_PROCESSED_DIR, PROJECT_ROOT, load_features


warnings.filterwarnings("ignore")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "model_a", "traditional")


def main(processed_dir=DEFAULT_PROCESSED_DIR, models_dir=MODELS_DIR):
    os.makedirs(models_dir, exist_ok=True)
    X_train, X_val, _X_test, y_train, y_val, _y_test = load_features(processed_dir)
    X_lex_train = X_train[:, -5:].toarray()
    X_lex_val = X_val[:, -5:].toarray()

    classifiers = [
        (
            "lr",
            LogisticRegression(max_iter=1000, C=1.0, solver="saga", n_jobs=-1),
            X_train,
            X_val,
        ),
        (
            "svm",
            CalibratedClassifierCV(LinearSVC(max_iter=2000), cv=3),
            X_train,
            X_val,
        ),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            ),
            X_lex_train,
            X_lex_val,
        ),
    ]

    for name, classifier, train_matrix, val_matrix in classifiers:
        start = time.time()
        classifier.fit(train_matrix, y_train)
        elapsed = time.time() - start
        predictions = classifier.predict(val_matrix)
        metrics = compute_metrics(y_val, predictions)
        print(f"{name}: acc={metrics['accuracy']:.4f}, f1={metrics['macro_f1']:.4f}, time={elapsed:.1f}s")
        joblib.dump(classifier, os.path.join(models_dir, f"{name}_model.pkl"))

    print("All Model A classifiers trained and saved.")


if __name__ == "__main__":
    main()
