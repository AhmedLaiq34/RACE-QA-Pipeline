"""Unified inference API for answer prediction, distractors, hints, and questions."""

import os
import re
import string
import warnings
from collections import Counter

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODELS_A = os.path.join(PROJECT_ROOT, "models", "model_a", "traditional")
MODELS_B = os.path.join(PROJECT_ROOT, "models", "model_b", "traditional")
OPTIONS = ("A", "B", "C", "D")
_MODELS = {}
WEAK_SUBJECT_STARTS = {
    "a",
    "an",
    "the",
    "this",
    "that",
    "these",
    "those",
    "he",
    "she",
    "it",
    "they",
    "we",
    "you",
    "i",
    "said",
}


def _load_models():
    if _MODELS:
        return
    _MODELS["ohe"] = joblib.load(os.path.join(MODELS_A, "ohe_vectorizer.pkl"))
    _MODELS["lr"] = joblib.load(os.path.join(MODELS_A, "lr_model.pkl"))
    _MODELS["svm"] = joblib.load(os.path.join(MODELS_A, "svm_model.pkl"))
    # RF model not used - only LR and SVM in weighted ensemble
    _MODELS["dist_vec"] = joblib.load(os.path.join(MODELS_B, "vectorizer_b.pkl"))
    _MODELS["dist_ranker"] = joblib.load(os.path.join(MODELS_B, "distractor_ranker.pkl"))
    _MODELS["hint_scorer"] = joblib.load(os.path.join(MODELS_B, "hint_scorer.pkl"))


def _clean(text):
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def _lexical_row(article, question, option):
    article_tokens = set(article.split())
    question_tokens = set(question.split())
    option_tokens = set(option.split())
    option_words = option.split()
    position = 0.0
    if option_words and option_words[0] in article:
        position = article.find(option_words[0]) / max(len(article), 1)
    return [
        len(option_words),
        len(question.split()),
        len(question_tokens & option_tokens),
        len(option_tokens & article_tokens),
        position,
    ]


def _answer_feature_matrix(article, question, options):
    ohe = _MODELS["ohe"]
    article_clean = _clean(article)
    question_clean = _clean(question)
    option_clean = [_clean(option) for option in options]
    combined = [f"{article_clean} [SEP] {question_clean} [SEP] {option}" for option in option_clean]

    ohe_matrix = ohe.transform(combined)
    shared_vectorizer = CountVectorizer(binary=True, vocabulary=ohe.vocabulary_)
    article_matrix = shared_vectorizer.transform([article_clean] * len(options))
    option_matrix = shared_vectorizer.transform(option_clean)
    cosine_values = [
        cosine_similarity(article_matrix[idx], option_matrix[idx])[0, 0] for idx in range(len(options))
    ]
    cosine_feature = csr_matrix(np.asarray(cosine_values, dtype=np.float32).reshape(-1, 1))
    lexical_feature = csr_matrix(
        np.asarray([_lexical_row(article_clean, question_clean, option) for option in option_clean], dtype=np.float32)
    )
    return hstack([ohe_matrix, cosine_feature, lexical_feature], format="csr"), lexical_feature.toarray()


def _positive_probability(model, matrix):
    probabilities = model.predict_proba(matrix)
    if probabilities.shape[1] == 1:
        return probabilities[:, 0]
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
        if 1 in classes:
            return probabilities[:, classes.index(1)]
    return probabilities[:, -1]


def predict_answer(article, question, options):
    """Return the predicted answer letter: A, B, C, or D.
    
    Uses weighted soft-vote ensemble: SVM × 0.6 + LR × 0.4
    """
    if len(options) != 4:
        raise ValueError("predict_answer expects exactly 4 options")
    _load_models()
    full_matrix, _lexical_matrix = _answer_feature_matrix(article, question, options)
    
    # Get probability scores from SVM and LR
    lr_scores = _positive_probability(_MODELS["lr"], full_matrix)
    svm_scores = _positive_probability(_MODELS["svm"], full_matrix)
    
    # Weighted ensemble: SVM × 0.6 + LR × 0.4 (SVM performs slightly better)
    scores = (svm_scores * 0.6) + (lr_scores * 0.4)
    
    return OPTIONS[int(np.argmax(scores))]


def _extract_ngram_candidates(article, answer, max_ngram=None):
    """Extract candidate distractors from article.
    
    Dynamically adjusts ngram size based on answer length to ensure
    distractors are similar in length to the answer.
    """
    tokens = _clean(article).split()
    answer_clean = _clean(answer)
    answer_length = len(answer_clean.split())
    
    # Dynamically set max_ngram based on answer length
    if max_ngram is None:
        if answer_length <= 2:
            max_ngram = 2
        elif answer_length <= 5:
            max_ngram = 4
        else:
            max_ngram = min(answer_length + 2, 8)  # Cap at 8 words
    
    candidates = []
    seen = set()
    stop = {"the", "a", "an", "is", "was", "are", "were", "of", "in", "to", "and", "or", "it", "them", "their", "they"}
    
    # Extract ngrams of various sizes, prioritizing similar length to answer
    for ngram_size in range(max(1, answer_length - 2), max_ngram + 1):
        for idx in range(len(tokens) - ngram_size + 1):
            candidate = " ".join(tokens[idx : idx + ngram_size])
            
            # Skip if too short, duplicate, or matches answer
            if candidate in seen or candidate == answer_clean or len(candidate) <= 2:
                continue
            
            # Skip if all stopwords
            if all(token in stop for token in candidate.split()):
                continue
            
            # Skip if overlaps with answer
            if candidate in answer_clean or answer_clean in candidate:
                continue
            
            # Skip single stopwords
            if ngram_size == 1 and candidate in stop:
                continue
            
            seen.add(candidate)
            candidates.append(candidate)
    
    return candidates


def _distractor_features(candidate, answer, article, vectorizer):
    candidate_vec = vectorizer.transform([candidate])
    answer_clean = _clean(answer)
    article_clean = _clean(article)
    answer_vec = vectorizer.transform([answer_clean])
    article_vec = vectorizer.transform([article_clean])
    article_tokens = article_clean.split()
    return [
        cosine_similarity(candidate_vec, answer_vec)[0, 0],
        cosine_similarity(candidate_vec, article_vec)[0, 0],
        article_tokens.count(candidate.split()[0]) / max(len(article_tokens), 1),
        sum(1 for a, b in zip(candidate, answer_clean) if a == b) / max(len(answer_clean), 1),
        len(candidate.split()) / max(len(answer_clean.split()), 1),
    ]


def generate_distractors(article, question, answer, n=3):
    """Generate n diverse distractor strings from the passage."""
    _load_models()
    vectorizer = _MODELS["dist_vec"]
    ranker = _MODELS["dist_ranker"]
    candidates = _extract_ngram_candidates(article, answer)
    if len(candidates) < n:
        stop = {"the", "a", "an", "is", "was", "are", "were", "of", "in", "to", "and", "or", "it"}
        answer_tokens = set(_clean(answer).split())
        for token, _count in Counter(_clean(article).split()).most_common(50):
            if token not in stop and token not in answer_tokens and token not in candidates and len(token) > 2:
                candidates.append(token)

    if not candidates:
        return ["Cannot be determined", "Not stated in the passage", "None of the above"][:n]

    limited_candidates = candidates[:120]
    feature_matrix = np.asarray(
        [_distractor_features(candidate, answer, article, vectorizer) for candidate in limited_candidates],
        dtype=np.float32,
    )
    scores = _positive_probability(ranker, feature_matrix)
    ranked = [limited_candidates[idx] for idx in np.argsort(scores)[::-1]]

    selected = []
    for candidate in ranked:
        if len(selected) >= n:
            break
        candidate_vec = vectorizer.transform([candidate])
        too_similar = any(
            cosine_similarity(candidate_vec, vectorizer.transform([existing]))[0, 0] > 0.8 for existing in selected
        )
        if not too_similar:
            selected.append(candidate)

    fallbacks = ["Cannot be determined", "Not stated in the passage", "None of the above", "All of the above"]
    for fallback in fallbacks:
        if len(selected) >= n:
            break
        if fallback not in selected and _clean(fallback) != _clean(answer):
            selected.append(fallback)
    return selected[:n]


def _hint_features(sentence, question, answer="", position=0, total_sentences=1):
    question_tokens = set(_clean(question).split())
    answer_tokens = set(_clean(answer).split())
    sentence_tokens = set(_clean(sentence).split())
    return [
        len(question_tokens & sentence_tokens) / max(len(question_tokens), 1),
        len(answer_tokens & sentence_tokens) / max(len(answer_tokens), 1) if answer_tokens else 0.0,
        position / max(total_sentences, 1),
        len(sentence.split()),
    ]


def get_hints(article, question, n=3):
    """Return n graduated hints from lower to higher model relevance."""
    _load_models()
    sentences = [sentence.strip() for sentence in re.split(r"[.!?]", str(article)) if len(sentence.strip()) > 10]
    if not sentences:
        return ["Refer back to the passage for more context."] * n

    features = np.asarray(
        [_hint_features(sentence, question, position=idx, total_sentences=len(sentences)) for idx, sentence in enumerate(sentences)],
        dtype=np.float32,
    )
    scores = _positive_probability(_MODELS["hint_scorer"], features)
    order = np.argsort(scores)
    if len(order) <= n:
        chosen_indices = list(order)
    else:
        positions = np.linspace(0, len(order) - 1, n).round().astype(int)
        chosen_indices = [int(order[pos]) for pos in positions]

    hints = []
    seen = set()
    for idx in chosen_indices:
        hint = sentences[idx]
        key = _clean(hint)
        if key and key not in seen:
            seen.add(key)
            hints.append(hint)
    while len(hints) < n:
        hints.append("Refer back to the passage for more context.")
    return hints[:n]


def _score_sentences(article, answer, top_k=3):
    sentences = [sentence.strip() for sentence in re.split(r"[.!?]", str(article)) if len(sentence.strip()) > 10]
    answer_tokens = set(_clean(answer).split())
    scored = []
    for sentence in sentences:
        sentence_tokens = set(_clean(sentence).split())
        overlap = len(sentence_tokens & answer_tokens) / max(len(answer_tokens), 1)
        scored.append((overlap, sentence))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [sentence for _score, sentence in scored[:top_k]]


def _question_templates(sentence, answer):
    candidates = []
    answer_clean = _clean(answer)
    sentence_clean = _clean(sentence)
    if answer_clean and answer_clean in sentence_clean:
        blanked = re.sub(re.escape(answer_clean), "___", sentence_clean, count=1)
        candidates.append(("fill_blank", f'Fill in the blank: "{blanked}"'))

    relation_match = re.search(r"\b(show|shows|showed)\b", sentence, flags=re.IGNORECASE)
    if relation_match:
        subject = sentence[: relation_match.start()].strip(" ,;:-")
        if 2 <= len(subject.split()) <= 12:
            subject = subject[0].lower() + subject[1:]
            aux = "do" if subject.split()[0].lower() in {"these", "those", "they"} else "does"
            candidates.append(("relation_show", f"What {aux} {subject} show?"))

    capitalized = re.findall(r"\b[A-Z][a-z]{2,}\b", sentence)
    capitalized = [word for word in capitalized if word.lower() not in WEAK_SUBJECT_STARTS]
    if capitalized:
        candidates.append(("who_what", f"Who or what is {capitalized[0]}?"))

    words = sentence.split()
    if len(words) >= 4:
        cleaned_words = [word.strip(string.punctuation) for word in words if word.strip(string.punctuation)]
        start = 0
        while start < len(cleaned_words) and cleaned_words[start].lower() in WEAK_SUBJECT_STARTS:
            start += 1
        subject_words = cleaned_words[start : start + 4]
        if subject_words and subject_words[0].lower() not in WEAK_SUBJECT_STARTS:
            subject = " ".join(subject_words)
            candidates.append(("generic_what", f"What can be said about {subject}?"))
    return candidates


def generate_question(article, answer):
    """Generate candidate questions for an article and answer."""
    results = []
    seen = set()
    for sentence in _score_sentences(article, answer, top_k=3):
        for template_name, question in _question_templates(sentence, answer):
            key = _clean(question)
            if key in seen:
                continue
            seen.add(key)
            results.append(
                {
                    "question": question,
                    "answer": answer,
                    "source_sentence": sentence,
                    "template": template_name,
                }
            )
    return results



def generate_question_auto(article):
    """Automatically generate question without requiring user to provide answer.
    
    This function:
    1. Extracts candidate answers from the article (entities, key terms)
    2. Generates questions for each candidate
    3. Ranks questions by quality
    4. Returns the best question with metadata
    
    Args:
        article: Article text
        
    Returns:
        dict with keys: question, answer, distractors, source_sentence, template
        Returns None if no valid question could be generated
    """
    # Extract candidate answers from article
    candidate_answers = _extract_candidate_answers(article)
    
    if not candidate_answers:
        return None
    
    # Generate questions for each candidate answer
    all_questions = []
    for answer in candidate_answers[:10]:  # Limit to top 10 candidates
        questions = generate_question(article, answer)
        for q in questions:
            q['candidate_answer'] = answer
            all_questions.append(q)
    
    if not all_questions:
        return None
    
    # Rank questions by quality and select best
    best_question = _rank_questions(all_questions, article)
    
    if not best_question:
        return None
    
    # Generate distractors for the best question
    answer = best_question['answer']
    question_text = best_question['question']
    distractors = generate_distractors(article, question_text, answer, n=3)
    
    return {
        'question': question_text,
        'answer': answer,
        'distractors': distractors,
        'source_sentence': best_question.get('source_sentence', ''),
        'template': best_question.get('template', 'auto')
    }


def _extract_candidate_answers(article):
    """Extract potential answers from article using multiple strategies.
    
    Strategies:
    1. Capitalized words (proper nouns - names, places)
    2. Numbers and dates
    3. Key noun phrases (frequent multi-word terms)
    4. Important single words (high frequency, not stopwords)
    
    Args:
        article: Article text
        
    Returns:
        List of candidate answer strings, ranked by likelihood
    """
    candidates = []
    seen = set()
    
    # Strategy 1: Capitalized words (proper nouns)
    capitalized = re.findall(r'\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)*\b', article)
    for cap in capitalized:
        cap_clean = cap.strip()
        if cap_clean and cap_clean.lower() not in WEAK_SUBJECT_STARTS and cap_clean not in seen:
            seen.add(cap_clean)
            candidates.append(cap_clean)
    
    # Strategy 2: Numbers and dates
    numbers = re.findall(r'\b\d{1,4}(?:,\d{3})*(?:\.\d+)?\b', article)
    for num in numbers:
        if num not in seen:
            seen.add(num)
            candidates.append(num)
    
    # Strategy 3: Key noun phrases (2-3 word phrases)
    words = article.split()
    for i in range(len(words) - 1):
        phrase = ' '.join(words[i:i+2])
        phrase_clean = _clean(phrase)
        # Check if phrase has at least one capitalized word or important term
        if (any(w[0].isupper() for w in words[i:i+2] if w) and 
            len(phrase_clean.split()) >= 2 and
            phrase not in seen):
            seen.add(phrase)
            candidates.append(phrase)
    
    # Strategy 4: Important single words (frequent, not stopwords)
    article_clean = _clean(article)
    word_freq = Counter(article_clean.split())
    stop_words = {'the', 'a', 'an', 'is', 'was', 'are', 'were', 'of', 'in', 'to', 'and', 
                  'or', 'it', 'for', 'on', 'with', 'as', 'by', 'at', 'from', 'this', 'that'}
    
    for word, freq in word_freq.most_common(20):
        if (word not in stop_words and 
            len(word) > 3 and 
            freq >= 2 and
            word not in seen):
            seen.add(word)
            candidates.append(word)
    
    return candidates[:15]  # Return top 15 candidates


def _rank_questions(questions, article):
    """Rank generated questions by quality metrics.
    
    Quality factors:
    1. Question length (prefer 8-20 words)
    2. Template diversity (prefer fill_blank and who_what)
    3. Answer presence in article (must be present)
    4. Question clarity (avoid very short or very long)
    
    Args:
        questions: List of question dicts
        article: Original article text
        
    Returns:
        Best question dict or None
    """
    if not questions:
        return None
    
    scored = []
    article_clean = _clean(article)
    
    for q in questions:
        score = 0
        question_text = q['question']
        answer = q['answer']
        template = q.get('template', '')
        
        # Factor 1: Question length (prefer 8-20 words)
        q_len = len(question_text.split())
        if 8 <= q_len <= 20:
            score += 3
        elif 5 <= q_len <= 25:
            score += 1
        
        # Factor 2: Template preference
        if template == 'fill_blank':
            score += 3  # Best template
        elif template in ['who_what', 'relation_show']:
            score += 2
        else:
            score += 1
        
        # Factor 3: Answer must be in article
        answer_clean = _clean(answer)
        if answer_clean in article_clean:
            score += 5
        else:
            score -= 10  # Heavy penalty
        
        # Factor 4: Answer length (prefer 1-4 words)
        ans_len = len(answer.split())
        if 1 <= ans_len <= 4:
            score += 2
        
        scored.append((score, q))
    
    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)
    
    # Return best question if score is positive
    if scored and scored[0][0] > 0:
        return scored[0][1]
    
    return None
