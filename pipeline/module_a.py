"""
Module A — Question & Answer Generation + Verification Pipeline

Loads trained models and exposes:
    generate_qa(article: str) -> dict       # Generate question + answer from article
    verify_answer(article, question, options) -> str  # Pick correct option from 4 choices
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
tfidf        = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")
lr_selector  = joblib.load(MODELS_DIR / "lr_sentence_selector.pkl")
nb_qtype     = joblib.load(MODELS_DIR / "nb_question_type.pkl")
count_vec    = joblib.load(MODELS_DIR / "count_vectorizer.pkl")

# Answer verifier (may not exist yet if notebook hasn't been re-run)
try:
    lr_verifier = joblib.load(MODELS_DIR / "lr_answer_verifier.pkl")
except FileNotFoundError:
    lr_verifier = None


def clean_tokens(text):
    text = re.sub(r'[^a-z0-9\s]', ' ', str(text).lower())
    return [t for t in text.split() if t not in STOP_WORDS and len(t) > 1]

def clean_str(text):
    return ' '.join(clean_tokens(text))


# ── Filler words to strip from questions ─────────────────────────────────
FILLERS = {'but', 'and', 'so', 'then', 'also', 'however', 'moreover',
           'therefore', 'thus', 'yet', 'still', 'besides', 'furthermore',
           'as', 'since', 'because', 'although', 'though', 'while'}


def _sentence_features(sent, target_text, sent_idx, total_sents):
    sent_tokens = clean_tokens(sent)
    target_tokens = set(clean_tokens(target_text))
    try:
        s_vec = tfidf.transform([sent])
        t_vec = tfidf.transform([target_text])
        tfidf_sim = cosine_similarity(s_vec, t_vec)[0, 0]
    except:
        tfidf_sim = 0.0
    length = len(sent_tokens)
    position = sent_idx / max(1, total_sents - 1)
    overlap = len(set(sent_tokens) & target_tokens) / max(1, length)
    words = sent.split()
    has_entity = int(any(w[0].isupper() and len(w) > 1 for w in words[1:] if w))
    return [tfidf_sim, length, position, overlap, has_entity]


def _select_sentence(article):
    sentences = sent_tokenize(str(article))
    if not sentences:
        return article[:200], sentences
    scored = []
    for idx, sent in enumerate(sentences):
        feats = _sentence_features(sent, article, idx, len(sentences))
        score = lr_selector.predict_proba([feats])[0][1]
        scored.append((sent, score, idx))
    scored.sort(key=lambda x: -x[1])
    return scored[0][0], sentences


def _classify_question_type(sentence, answer_hint=""):
    combined = f"[ANS] {clean_str(answer_hint)} [SENT] {clean_str(sentence)}"
    vec = count_vec.transform([combined])
    return nb_qtype.predict(vec)[0]


WH_MAP = {
    'what': 'What', 'who': 'Who', 'where': 'Where',
    'when': 'When', 'why': 'Why', 'how': 'How', 'which': 'Which'
}

def _build_question(sentence, wh_type):
    wh = WH_MAP.get(wh_type, 'What')
    cleaned = sentence.strip().rstrip('.!,;:')
    words = cleaned.split()
    # Remove leading filler words
    while words and words[0].lower().strip('.,') in FILLERS:
        words = words[1:]
    # Limit to 12 words
    if len(words) > 12:
        words = words[:12]
    cleaned = ' '.join(words)
    return f"{wh} {cleaned.lower()}?"


def generate_qa(article):
    """
    Full Module A pipeline:
    1. Select best sentence (LR)
    2. Classify question type (NB)
    3. Generate question (template)
    4. Answer = source sentence
    """
    source_sentence, all_sentences = _select_sentence(article)
    q_type = _classify_question_type(source_sentence)
    question = _build_question(source_sentence, q_type)
    answer = source_sentence
    return {
        'source_sentence': source_sentence,
        'question_type': q_type,
        'question': question,
        'answer': answer,
    }


# ── Answer Verification ──────────────────────────────────────────────────

def _verifier_features(article, question, option):
    """7 features for answer verification."""
    art_tokens = set(clean_tokens(article))
    q_tokens = set(clean_tokens(question))
    opt_tokens = set(clean_tokens(option))

    try:
        aq_vec = tfidf.transform([article + " " + question])
        opt_vec = tfidf.transform([option])
        sim_aq_opt = cosine_similarity(aq_vec, opt_vec)[0, 0]
    except:
        sim_aq_opt = 0.0

    try:
        a_vec = tfidf.transform([article])
        sim_a_opt = cosine_similarity(a_vec, opt_vec)[0, 0]
    except:
        sim_a_opt = 0.0

    q_overlap = len(q_tokens & opt_tokens) / max(1, len(opt_tokens))
    a_overlap = len(art_tokens & opt_tokens) / max(1, len(opt_tokens))
    opt_len = len(opt_tokens)
    aq_tokens = art_tokens | q_tokens
    jaccard = len(aq_tokens & opt_tokens) / max(1, len(aq_tokens | opt_tokens))
    novel_ratio = len(opt_tokens - art_tokens) / max(1, len(opt_tokens))

    return [sim_aq_opt, sim_a_opt, q_overlap, a_overlap, opt_len, jaccard, novel_ratio]


def verify_answer(article, question, options):
    """
    Given an article, question, and dict of options {A: text, B: text, ...},
    return the predicted correct option label.
    """
    if lr_verifier is None:
        # Fallback: TF-IDF cosine similarity
        aq_vec = tfidf.transform([article + " " + question])
        best_label, best_sim = 'A', -1
        for label, text in options.items():
            opt_vec = tfidf.transform([text])
            sim = cosine_similarity(aq_vec, opt_vec)[0, 0]
            if sim > best_sim:
                best_sim = sim
                best_label = label
        return best_label

    best_label, best_prob = 'A', -1
    for label, text in options.items():
        feats = _verifier_features(article, question, text)
        prob = lr_verifier.predict_proba([feats])[0][1]
        if prob > best_prob:
            best_prob = prob
            best_label = label

    return best_label
