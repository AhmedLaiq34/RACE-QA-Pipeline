"""Preprocessing utilities for the RACE QA pipeline."""

import os
import re
import string

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, load_npz, save_npz
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DEFAULT_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DEFAULT_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
DEFAULT_MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "model_a", "traditional")
OPTIONS = ("A", "B", "C", "D")


def _resolve_path(path_value):
    if path_value is None:
        return None
    if os.path.isabs(str(path_value)):
        return str(path_value)
    return os.path.join(PROJECT_ROOT, str(path_value))


def clean_text(text):
    """Lowercase, remove punctuation, and collapse whitespace."""
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def prepare_text_columns(df):
    """Return a copy with cleaned article, question, and option columns."""
    result = df.copy()
    result["article_clean"] = result["article"].apply(clean_text)
    result["question_clean"] = result["question"].apply(clean_text)
    for option in OPTIONS:
        result[f"{option}_clean"] = result[option].apply(clean_text)
    return result


def expand_df(df):
    """Expand each MCQ row into four binary verification rows."""
    rows = []
    for row in df.itertuples(index=False):
        row_dict = row._asdict()
        article_clean = row_dict.get("article_clean", clean_text(row_dict["article"]))
        question_clean = row_dict.get("question_clean", clean_text(row_dict["question"]))
        answer_letter = str(row_dict["answer"]).strip()

        for option in OPTIONS:
            option_clean = row_dict.get(f"{option}_clean", clean_text(row_dict[option]))
            rows.append(
                {
                    "article": article_clean,
                    "question": question_clean,
                    "option": option_clean,
                    "option_letter": option,
                    "label": 1 if answer_letter == option else 0,
                    "combined_text": f"{article_clean} [SEP] {question_clean} [SEP] {option_clean}",
                    "article_raw": row_dict["article"],
                    "question_raw": row_dict["question"],
                    "A_raw": row_dict["A"],
                    "B_raw": row_dict["B"],
                    "C_raw": row_dict["C"],
                    "D_raw": row_dict["D"],
                    "answer": answer_letter,
                }
            )
    return pd.DataFrame(rows)


def _rowwise_cosine(article_texts, option_texts, vocabulary):
    vectorizer = CountVectorizer(binary=True, vocabulary=vocabulary)
    article_matrix = vectorizer.transform(article_texts)
    option_matrix = vectorizer.transform(option_texts)
    numerator = np.asarray(article_matrix.multiply(option_matrix).sum(axis=1)).ravel()
    article_norm = np.sqrt(np.asarray(article_matrix.multiply(article_matrix).sum(axis=1)).ravel())
    option_norm = np.sqrt(np.asarray(option_matrix.multiply(option_matrix).sum(axis=1)).ravel())
    denominator = article_norm * option_norm
    values = np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=np.float32), where=denominator != 0)
    return csr_matrix(values.reshape(-1, 1))


def _lexical_features(df_expanded):
    rows = []
    for article, question, option in zip(df_expanded["article"], df_expanded["question"], df_expanded["option"]):
        article = str(article)
        question = str(question)
        option = str(option)
        article_tokens = set(article.split())
        question_tokens = set(question.split())
        option_tokens = set(option.split())
        option_words = option.split()

        position = 0.0
        if option_words and option_words[0] in article:
            position = article.find(option_words[0]) / max(len(article), 1)

        rows.append(
            [
                len(option_words),
                len(question.split()),
                len(question_tokens & option_tokens),
                len(option_tokens & article_tokens),
                position,
            ]
        )
    return csr_matrix(np.asarray(rows, dtype=np.float32))


def build_features(
    train_exp,
    val_exp,
    test_exp,
    save_dir=DEFAULT_PROCESSED_DIR,
    models_dir=DEFAULT_MODELS_DIR,
    max_features=5000,
):
    """Build and save OHE + cosine + lexical feature matrices."""
    save_dir = _resolve_path(save_dir)
    models_dir = _resolve_path(models_dir)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    vectorizer = CountVectorizer(binary=True, max_features=max_features, min_df=2)
    train_ohe = vectorizer.fit_transform(train_exp["combined_text"])
    val_ohe = vectorizer.transform(val_exp["combined_text"])
    test_ohe = vectorizer.transform(test_exp["combined_text"])
    joblib.dump(vectorizer, os.path.join(models_dir, "ohe_vectorizer.pkl"))

    for split_name, expanded_df, sparse_ohe in (
        ("train", train_exp, train_ohe),
        ("val", val_exp, val_ohe),
        ("test", test_exp, test_ohe),
    ):
        full_matrix = hstack(
            [
                sparse_ohe,
                _rowwise_cosine(expanded_df["article"], expanded_df["option"], vectorizer.vocabulary_),
                _lexical_features(expanded_df),
            ],
            format="csr",
        )
        save_npz(os.path.join(save_dir, f"X_{split_name}.npz"), full_matrix)
        np.save(os.path.join(save_dir, f"y_{split_name}.npy"), expanded_df["label"].to_numpy())

    return vectorizer


def load_raw_splits(raw_dir=DEFAULT_RAW_DIR):
    """Load train, validation, and test CSV files."""
    raw_dir = _resolve_path(raw_dir)
    train_df = pd.read_csv(os.path.join(raw_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(raw_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(raw_dir, "test.csv"))
    return train_df, val_df, test_df


def save_expanded_splits(train_exp, val_exp, test_exp, save_dir=DEFAULT_PROCESSED_DIR):
    """Persist expanded splits for debugging and CLI random sampling."""
    save_dir = _resolve_path(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    train_exp.to_csv(os.path.join(save_dir, "train_exp.csv"), index=False)
    val_exp.to_csv(os.path.join(save_dir, "val_exp.csv"), index=False)
    test_exp.to_csv(os.path.join(save_dir, "test_exp.csv"), index=False)


def preprocess_and_build(raw_dir=DEFAULT_RAW_DIR, save_dir=DEFAULT_PROCESSED_DIR, models_dir=DEFAULT_MODELS_DIR):
    """Run the complete raw CSV to saved features pipeline."""
    train_df, val_df, test_df = load_raw_splits(raw_dir)
    train_exp = expand_df(prepare_text_columns(train_df))
    val_exp = expand_df(prepare_text_columns(val_df))
    test_exp = expand_df(prepare_text_columns(test_df))
    save_expanded_splits(train_exp, val_exp, test_exp, save_dir)
    build_features(train_exp, val_exp, test_exp, save_dir, models_dir)
    return train_exp, val_exp, test_exp


def load_features(processed_dir=DEFAULT_PROCESSED_DIR):
    """Load saved sparse matrices and labels."""
    processed_dir = _resolve_path(processed_dir)
    X_train = load_npz(os.path.join(processed_dir, "X_train.npz"))
    X_val = load_npz(os.path.join(processed_dir, "X_val.npz"))
    X_test = load_npz(os.path.join(processed_dir, "X_test.npz"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    y_val = np.load(os.path.join(processed_dir, "y_val.npy"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))
    return X_train, X_val, X_test, y_train, y_val, y_test


__all__ = [
    "build_features",
    "clean_text",
    "expand_df",
    "load_features",
    "load_raw_splits",
    "prepare_text_columns",
    "preprocess_and_build",
    "save_expanded_splits",
]
