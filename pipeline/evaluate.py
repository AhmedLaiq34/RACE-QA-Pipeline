"""Evaluation helpers for classification and text generation."""

import re
import string

import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def compute_metrics(y_true, y_pred, y_proba=None, n_options=4):
    """Compute binary verification metrics, plus MCQ exact match when probabilities are supplied."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_proba is not None:
        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)
        n_groups = len(y_true) // n_options
        correct = 0
        counted = 0
        for group_idx in range(n_groups):
            start = group_idx * n_options
            end = start + n_options
            true_group = y_true[start:end]
            proba_group = y_proba[start:end]
            if len(true_group) == n_options and true_group.sum() > 0:
                correct += int(np.argmax(proba_group) == np.argmax(true_group))
                counted += 1
        metrics["exact_match"] = float(correct / max(counted, 1))
    return metrics


def _clean_for_eval(text):
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def compute_generation_metrics(pred_texts, ref_texts):
    """Compute average BLEU, ROUGE-1, ROUGE-L, and METEOR for paired texts."""
    if len(pred_texts) != len(ref_texts):
        raise ValueError("pred_texts and ref_texts must have the same length")
    if not pred_texts:
        return {"bleu": 0.0, "rouge_1": 0.0, "rouge_l": 0.0, "meteor": 0.0}

    smooth = SmoothingFunction().method1
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    bleu_values = []
    rouge_1_values = []
    rouge_l_values = []
    meteor_values = []

    for pred, ref in zip(pred_texts, ref_texts):
        pred_clean = _clean_for_eval(pred)
        ref_clean = _clean_for_eval(ref)
        pred_tokens = pred_clean.split()
        ref_tokens = ref_clean.split()

        bleu_values.append(
            float(sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth))
            if pred_tokens and ref_tokens
            else 0.0
        )
        rouge = scorer.score(ref_clean, pred_clean)
        rouge_1_values.append(float(rouge["rouge1"].fmeasure))
        rouge_l_values.append(float(rouge["rougeL"].fmeasure))
        meteor_values.append(float(meteor_score([ref_tokens], pred_tokens)) if pred_tokens and ref_tokens else 0.0)

    return {
        "bleu": float(np.mean(bleu_values)),
        "rouge_1": float(np.mean(rouge_1_values)),
        "rouge_l": float(np.mean(rouge_l_values)),
        "meteor": float(np.mean(meteor_values)),
    }
