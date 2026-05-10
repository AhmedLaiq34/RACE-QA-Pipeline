"""Train Model B distractor and hint scoring models."""

import os
import re
import warnings
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

try:
    from .preprocessing import DEFAULT_RAW_DIR, PROJECT_ROOT, clean_text
except ImportError:
    from preprocessing import DEFAULT_RAW_DIR, PROJECT_ROOT, clean_text


warnings.filterwarnings("ignore")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "model_b", "traditional")
OPTIONS = ("A", "B", "C", "D")


def extract_candidates(article, answer, max_ngram=2, limit=None):
    tokens = clean_text(article).split()
    answer_clean = clean_text(answer)
    candidates = []
    seen = set()
    for ngram_size in range(1, max_ngram + 1):
        for idx in range(len(tokens) - ngram_size + 1):
            candidate = " ".join(tokens[idx : idx + ngram_size])
            if candidate == answer_clean or len(candidate) <= 2 or candidate in seen:
                continue
            seen.add(candidate)
            candidates.append(candidate)
            if limit and len(candidates) >= limit:
                return candidates
    return candidates


def _char_match_score(candidate, answer):
    return sum(1 for a, b in zip(candidate, answer) if a == b) / max(len(answer), 1)


def _distractor_features(candidate, answer, article, vectorizer):
    candidate_vec = vectorizer.transform([candidate])
    answer_clean = clean_text(answer)
    article_clean = clean_text(article)
    answer_vec = vectorizer.transform([answer_clean])
    article_vec = vectorizer.transform([article_clean])
    article_tokens = article_clean.split()
    return [
        cosine_similarity(candidate_vec, answer_vec)[0, 0],
        cosine_similarity(candidate_vec, article_vec)[0, 0],
        article_tokens.count(candidate.split()[0]) / max(len(article_tokens), 1),
        _char_match_score(candidate, answer_clean),
        len(candidate.split()) / max(len(answer_clean.split()), 1),
    ]


def _hint_features(sentence, question, answer, position, total_sentences):
    question_tokens = set(clean_text(question).split())
    answer_tokens = set(clean_text(answer).split())
    sentence_tokens = set(clean_text(sentence).split())
    answer_overlap = len(answer_tokens & sentence_tokens) / max(len(answer_tokens), 1)
    return [
        len(question_tokens & sentence_tokens) / max(len(question_tokens), 1),
        answer_overlap,
        position / max(total_sentences, 1),
        len(sentence.split()),
    ], answer_overlap


def main(raw_dir=DEFAULT_RAW_DIR, models_dir=MODELS_DIR, sample_size=500):
    os.makedirs(models_dir, exist_ok=True)
    train_df = pd.read_csv(os.path.join(raw_dir, "train.csv"))

    vectorizer = CountVectorizer(binary=True, max_features=3000)
    vectorizer.fit(train_df["article"].apply(clean_text).tolist())
    joblib.dump(vectorizer, os.path.join(models_dir, "vectorizer_b.pkl"))

    sample_n = min(sample_size, len(train_df))
    sampled = train_df.sample(sample_n, random_state=42)

    distractor_x = []
    distractor_y = []
    for _, row in sampled.iterrows():
        correct_answer = row[row["answer"]]
        gold_distractors = [clean_text(row[option]) for option in OPTIONS if option != row["answer"]]
        for candidate in extract_candidates(row["article"], correct_answer, limit=30):
            distractor_x.append(_distractor_features(candidate, correct_answer, row["article"], vectorizer))
            distractor_y.append(int(any(candidate in gold or gold in candidate for gold in gold_distractors)))

    distractor_x = np.asarray(distractor_x, dtype=np.float32)
    distractor_y = np.asarray(distractor_y, dtype=np.int64)
    if len(set(distractor_y.tolist())) < 2:
        distractor_y[0] = 1 - distractor_y[0]
    distractor_ranker = LogisticRegression(max_iter=500)
    distractor_ranker.fit(distractor_x, distractor_y)
    joblib.dump(distractor_ranker, os.path.join(models_dir, "distractor_ranker.pkl"))
    print(f"Distractor ranker trained. Acc: {accuracy_score(distractor_y, distractor_ranker.predict(distractor_x)):.4f}")

    hint_x = []
    hint_y = []
    for _, row in sampled.iterrows():
        answer = row[row["answer"]]
        sentences = [sentence.strip() for sentence in re.split(r"[.!?]", str(row["article"])) if len(sentence.strip()) > 15]
        for pos, sentence in enumerate(sentences[:15]):
            features, answer_overlap = _hint_features(sentence, row["question"], answer, pos, len(sentences))
            hint_x.append(features)
            hint_y.append(1 if answer_overlap > 0.3 else 0)

    hint_x = np.asarray(hint_x, dtype=np.float32)
    hint_y = np.asarray(hint_y, dtype=np.int64)
    if len(set(hint_y.tolist())) < 2:
        hint_y[0] = 1 - hint_y[0]
    hint_scorer = LogisticRegression(max_iter=500)
    hint_scorer.fit(hint_x, hint_y)
    joblib.dump(hint_scorer, os.path.join(models_dir, "hint_scorer.pkl"))
    print(f"Hint scorer trained. Acc: {accuracy_score(hint_y, hint_scorer.predict(hint_x)):.4f}")


if __name__ == "__main__":
    main()
