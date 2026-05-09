"""
Module B — Distractor Generation Pipeline

Loads trained models and exposes a single function:
    generate_distractors(article, question, correct_answer, n=3) -> list[str]

Returns a list of 3 plausible distractor sentences.
"""

import re
import joblib
import numpy as np
from pathlib import Path
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity

STOP_WORDS = set(stopwords.words('english'))
MODELS_DIR = Path(__file__).parent.parent / "models"

# ── Load trained models ──────────────────────────────────────────────────
tfidf          = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")
rf_distractor  = joblib.load(MODELS_DIR / "rf_distractor_ranker.pkl")


def clean_tokens(text):
    text = re.sub(r'[^a-z0-9\s]', ' ', str(text).lower())
    return [t for t in text.split() if t not in STOP_WORDS and len(t) > 1]


def _distractor_features(candidate, correct_answer, article):
    """Compute features for a candidate distractor."""
    cand_tokens = clean_tokens(candidate)
    ans_tokens = clean_tokens(correct_answer)

    try:
        c_vec = tfidf.transform([candidate])
        a_vec = tfidf.transform([correct_answer])
        sim_to_answer = cosine_similarity(c_vec, a_vec)[0, 0]
    except:
        sim_to_answer = 0.0

    try:
        art_vec = tfidf.transform([article])
        sim_to_article = cosine_similarity(c_vec, art_vec)[0, 0]
    except:
        sim_to_article = 0.0

    len_ratio = len(cand_tokens) / max(1, len(ans_tokens))
    overlap = len(set(cand_tokens) & set(ans_tokens)) / max(1, len(cand_tokens))
    char_ratio = len(candidate) / max(1, len(correct_answer))

    return [sim_to_answer, sim_to_article, len_ratio, overlap, char_ratio]


def _extract_candidates(article, correct_answer):
    """
    Extract candidate distractor phrases from the passage.
    
    Strategy:
    1. Split article into sentences
    2. Filter out the correct answer sentence
    3. Also generate sub-phrases from longer sentences
    """
    sentences = sent_tokenize(str(article))
    candidates = []

    ans_lower = correct_answer.lower().strip()

    for sent in sentences:
        sent = sent.strip()
        # Skip if it's too similar to the correct answer
        if sent.lower().strip() == ans_lower:
            continue
        if len(sent.split()) < 3:
            continue

        candidates.append(sent)

        # Also extract sub-clauses (split on commas/semicolons)
        parts = re.split(r'[,;]', sent)
        for part in parts:
            part = part.strip()
            if len(part.split()) >= 3 and part.lower() != ans_lower:
                candidates.append(part)

    # Deduplicate
    seen = set()
    unique = []
    for c in candidates:
        c_lower = c.lower().strip()
        if c_lower not in seen:
            seen.add(c_lower)
            unique.append(c)

    return unique


def generate_distractors(article, question, correct_answer, n=3):
    """
    Full Module B pipeline:
    1. Extract candidate phrases from the passage
    2. Compute features for each candidate
    3. Score with RF ranker
    4. Return top-N candidates as distractors

    Returns list of n distractor strings.
    """
    # Step 1: Extract candidates
    candidates = _extract_candidates(article, correct_answer)

    if len(candidates) == 0:
        return ["No distractor available."] * n

    # Step 2: Compute features and score
    scored = []
    for cand in candidates:
        feats = _distractor_features(cand, correct_answer, article)
        # Get probability of being a good distractor
        prob = rf_distractor.predict_proba([feats])[0][1]
        scored.append((cand, prob))

    # Step 3: Sort by score (highest = best distractor)
    scored.sort(key=lambda x: -x[1])

    # Step 4: Return top-N
    distractors = [s[0] for s in scored[:n]]

    # Pad if not enough candidates
    while len(distractors) < n:
        distractors.append("Not enough candidates in passage.")

    return distractors
