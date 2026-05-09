import os
import random
import re
import warnings
import numpy as np
import pandas as pd
import joblib
import nltk
from nltk.tokenize import sent_tokenize
from scipy.sparse import hstack
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
STOP_WORDS = set(stopwords.words('english'))

print("Initializing NLP Pipeline & Loading Models...")

MODELS_DIR = Path("models_new")
PROCESSED_DIR = Path("data/processed")

try:
    tfidf = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")
    verifier = joblib.load(MODELS_DIR / "lr_balanced.pkl")
    dist_ranker = joblib.load(MODELS_DIR / "module_b_rf.pkl")
except FileNotFoundError:
    print("Error: Models not found in 'models_new/'.")
    exit(1)

# ---------------------------------------------------------
# NEW LOGIC: Section 8.2 and 8.3 Implementation
# ---------------------------------------------------------
def char_jaccard(str1, str2):
    set1 = set(str1.lower())
    set2 = set(str2.lower())
    union = len(set1 | set2)
    if union == 0: return 0.0
    return len(set1 & set2) / union

def get_ngrams(text, n=2):
    tokens = nltk.word_tokenize(str(text))
    ngrams = []
    for i in range(len(tokens)-n+1):
        ngrams.append(" ".join(tokens[i:i+n]))
    return ngrams

def enumerate_candidate_questions(article, top_k=3):
    """Generates multiple candidates via localized TF-IDF and specialized templates."""
    sentences = sent_tokenize(str(article))
    if not sentences:
        sentences = [str(article)]
        
    # 1. Passage-level TF-IDF
    try:
        tfidf_local = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1,2), sublinear_tf=True)
        mat = tfidf_local.fit_transform(sentences)
        feature_names = tfidf_local.get_feature_names_out()
    except ValueError:
        mat = None
        feature_names = []
        
    scored_sentences = []
    
    # 2. Sentence Salience Scoring
    for i, sent in enumerate(sentences):
        sent = sent.replace('\n', ' ').replace('\r', '')
        words = sent.split()
        if len(words) < 4: continue
            
        score = 0.0
        max_term = ""
        
        if mat is not None and mat.shape[0] > i:
            row = mat.getrow(i)
            if row.nnz > 0:
                max_idx = row.indices[row.data.argmax()]
                score = row.data.max()  # Base score
                max_term = feature_names[max_idx]
                
        sent_lower = sent.lower()
        
        # Location bias (+0.05)
        if any(cue in sent_lower for cue in ["located", "capital", "city"]):
            score += 0.05
            
        # Preferred-Candidate Shortlist
        is_shortlisted = False
        if any(pat in sent_lower for pat in ["located", "in ", "capital of"]):
            is_shortlisted = True
            score += 0.5 # Boost
            
        scored_sentences.append((score, is_shortlisted, len(words), i, sent, max_term))
        
    # Sort: Shortlist first, then score, then length as fallback
    scored_sentences.sort(key=lambda x: (x[1], x[0], x[2]), reverse=True)
    
    candidates = []
    seen = set()
    
    for _, _, _, _, sent, max_term in scored_sentences:
        if len(candidates) >= top_k: break
            
        answer = ""
        
        # 3. Answer Phrase Extraction (Aggressively prefer Entities for Who/Where/When)
        answer = ""
        match_loc = re.search(r"is located in ([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)", sent)
        match_year = re.search(r"(19|20)\d{2}", sent)
        
        # Look for proper nouns (capitalized words not at the start of sentence)
        words_original = sent.split()
        capitalized = [w.strip('.,?!;"\'') for w in words_original[1:] if w.strip('.,?!;"\'').istitle() and len(w.strip('.,?!;"\'')) > 2]
        
        if match_loc:
            answer = match_loc.group(1)
        elif match_year:
            answer = match_year.group(0)
        elif capitalized and any(c in sent_lower.split() for c in ["president", "author", "scientist", "he", "she", "man", "woman", "boy", "girl", "person", "people", "student", "teacher"]):
            # If it's a person context and we have a capitalized word, assume it's a name!
            answer = capitalized[-1]
        elif capitalized and (any(c in sent_lower.split() for c in ["city", "country", "town", "park", "school", "house", "street", "museum"]) or "in " in sent_lower):
            # Location context + capitalized word
            answer = capitalized[-1]
        elif max_term:
            answer = max_term
        else: # Sparse-Row No-Signal Fallback
            tokens = [w for w in re.findall(r"\w+", sent) if w.lower() not in STOP_WORDS]
            answer = max(tokens, key=len) if tokens else sent.split()[-1]
            
        # 4. Mask Construction
        mask_idx = sent.find(answer)
        if mask_idx != -1:
            masked = sent[:mask_idx] + "_____" + sent[mask_idx+len(answer):]
        else:
            # Case-insensitive fallback
            lower_idx = sent.lower().find(answer.lower())
            if lower_idx != -1:
                masked = sent[:lower_idx] + "_____" + sent[lower_idx+len(answer):]
            else:
                masked = sent
                
        # 5. Template Routing (Improved logic focusing on the answer type)
        sent_lower = sent.lower()
        ans_lower = answer.lower()
        q_type = "What"
        template_name = "cloze"
        
        # WHEN: Is the answer a year or time?
        if re.search(r"(19|20)\d{2}", answer) or any(c in ans_lower.split() for c in ["year", "date", "month", "day", "time", "today", "yesterday", "tomorrow", "morning", "night"]):
            q_type, template_name = "When", "when"
            
        # WHERE: Is the answer a place, or is it preceded by a location preposition?
        elif any(c in ans_lower.split() for c in ["city", "country", "town", "park", "school", "house", "street", "museum", "uk", "usa"]) or f"in {ans_lower}" in sent_lower or f"at {ans_lower}" in sent_lower or f"to {ans_lower}" in sent_lower:
            q_type, template_name = "Where", "where"
            
        # WHO: Is the answer a person?
        elif any(c in ans_lower.split() for c in ["president", "author", "scientist", "he", "she", "man", "woman", "boy", "girl", "person", "people", "student", "teacher", "they", "kids", "children"]):
            q_type, template_name = "Who", "who"
            
        if template_name == "cloze":
            question = f"According to the passage, what best completes this sentence: {masked}"
        else:
            question = f"Based on the passage, {q_type.lower()} best completes this sentence: {masked}?"
            
        key = (question.lower(), answer.lower())
        if key not in seen:
            seen.add(key)
            candidates.append({
                'question': question,
                'correct_answer': answer,
                'source_sentence': sent,
                'template': template_name
            })
            
    return candidates

def get_distractors(article, correct_answer):
    article_str = str(article).lower().replace('\n', ' ').replace('\r', '')
    unigrams = article_str.split()
    bigrams = get_ngrams(article_str, 2)
    trigrams = get_ngrams(article_str, 3)
    
    all_cands = unigrams + bigrams + trigrams
    freq = {}
    for c in all_cands:
        if len(c) > 2:
            freq[c] = freq.get(c, 0) + 1
            
    candidates = list(freq.keys())
    # Filter
    ans_lower = str(correct_answer).lower()
    
    filtered_cands = []
    for c in candidates:
        if c in ans_lower or ans_lower in c: continue
        
        # Aggressively clean punctuation to check stopwords
        c_clean = re.sub(r'[^\w\s]', '', c).strip()
        
        if len(c_clean) <= 2: continue
        if c_clean in STOP_WORDS: continue
        
        # Ignore if the cleaned version is entirely stopwords
        if all(w in STOP_WORDS for w in c_clean.split()): continue
        
        # Ignore if it doesn't have actual letters
        if not re.search('[a-z]', c_clean): continue
        
        # Don't use strings with carriage returns or newlines
        if '\n' in c or '\r' in c: continue
        
        filtered_cands.append(c.replace('\n', ' ').replace('\r', ''))
        
    candidates = filtered_cands
    
    if len(candidates) < 3:
        return ["None of the above", "All of the above", "Cannot be determined"]
        
    ans_mat = tfidf.transform([str(correct_answer)])
    cand_mat = tfidf.transform(candidates)
    sim_scores = ans_mat.multiply(cand_mat).sum(axis=1).A1
    
    features = []
    for i, cand in enumerate(candidates):
        sim = sim_scores[i]
        cmatch = char_jaccard(str(correct_answer), cand)
        f = freq[cand]
        features.append([sim, cmatch, f])
        
    features = np.array(features)
    probs = dist_ranker.predict_proba(features)[:, 1]
    
    top_indices = np.argsort(probs)[::-1][:3]
    return [candidates[i] for i in top_indices]

def extract_features_for_row(article, question, option, tfidf_model):
    combined = article + " [SEP] " + question + " [SEP] " + option
    tf_vec = tfidf_model.transform([combined])
    
    art_tokens = set(article.split())
    q_tokens = set(question.split())
    opt_tokens = set(option.split())
    
    a_vec = tfidf_model.transform([article])
    q_vec = tfidf_model.transform([question])
    o_vec = tfidf_model.transform([option])
    
    sim_aq = cosine_similarity(a_vec, q_vec)[0,0]
    sim_ao = cosine_similarity(a_vec, o_vec)[0,0]
    sim_qo = cosine_similarity(q_vec, o_vec)[0,0]
    
    sentences = sent_tokenize(article)
    best_sim, sim_a_best, sim_o_best, sim_q_best = 0.0, 0.0, 0.0, 0.0
    if sentences:
        s_vecs = tfidf_model.transform(sentences)
        sims = cosine_similarity(q_vec, s_vecs)[0]
        best_idx = np.argmax(sims)
        best_sent_vec = s_vecs[best_idx].reshape(1, -1)
        sim_a_best = cosine_similarity(a_vec, best_sent_vec)[0,0]
        sim_o_best = cosine_similarity(o_vec, best_sent_vec)[0,0]
        sim_q_best = sims[best_idx]
        
    len_a, len_q, len_o = len(art_tokens), len(q_tokens), len(opt_tokens)
    overlap_qo = len(q_tokens & opt_tokens) / max(1, len_o)
    overlap_ao = len(art_tokens & opt_tokens) / max(1, len_o)
    exact_match = 1 if option in article else 0
    freq = article.count(option)
    
    num_vec = np.array([[sim_aq, sim_ao, sim_qo, sim_a_best, sim_o_best, sim_q_best,
                         len_a, len_q, len_o, overlap_qo, overlap_ao, exact_match, freq]])
    return hstack([tf_vec, num_vec])

def run_app():
    print("\n" + "="*50)
    print("  RACE QA Generation & Verification CLI v2.0")
    print("="*50)
    
    while True:
        print("\nOptions:")
        print("  1. Enter your own article")
        print("  2. Load a random RACE article")
        print("  3. Exit")
        choice = input("Select an option (1/2/3): ").strip()
        
        if choice == '3': break
            
        article = ""
        if choice == '1':
            article = input("\nPaste your article text below:\n> ").strip()
        elif choice == '2':
            try:
                val_df = pd.read_csv(PROCESSED_DIR / "val_verification.csv")
                article = random.choice(val_df['article'].dropna().unique())
                print("\nLoaded random article:")
                print(article[:400] + "...\n")
            except Exception as e:
                print("Error loading dataset.")
                continue
        else: continue
            
        if not article or len(article.split()) < 4:
            print("Article is too short.")
            continue

        print("\n[Processing...] Generating up to 5 Candidate Questions...")
        candidates = enumerate_candidate_questions(article, top_k=5)
        
        if not candidates:
            print("Failed to generate candidates.")
            continue
            
        print("[Processing...] Using ML Verifier to rank the best question...")
        
        best_overall_score = -np.inf
        best_cand = None
        best_options = None
        best_scores = None
        
        # ML Ranking Loop
        for cand in candidates:
            q_text = cand['question']
            c_ans = cand['correct_answer']
            
            dists = get_distractors(article, c_ans)
            opts = [c_ans] + dists
            random.shuffle(opts)
            
            scores = []
            for opt in opts:
                feats = extract_features_for_row(article, q_text, opt, tfidf)
                scores.append(verifier.predict_proba(feats)[0, 1])
                
            pred_idx = np.argmax(scores)
            pred_opt = opts[pred_idx]
            
            # We only keep the question if the ML verifier gets the answer right.
            # This ensures we don't present a confusing question.
            if pred_opt == c_ans:
                if scores[pred_idx] > best_overall_score:
                    best_overall_score = scores[pred_idx]
                    best_cand = cand
                    best_options = opts
                    best_scores = scores
                    
        # Fallback if the verifier failed on all candidates
        if best_cand is None:
            best_cand = candidates[0]
            c_ans = best_cand['correct_answer']
            dists = get_distractors(article, c_ans)
            best_options = [c_ans] + dists
            random.shuffle(best_options)
            
        cand = candidates[0]
        question = cand['question']
        correct_answer = cand['correct_answer']
        
        distractors = get_distractors(article, correct_answer)
        options = [correct_answer] + distractors
        random.shuffle(options)
        correct_letter = chr(65 + options.index(correct_answer))

        print("\n--------------------------------------------------")
        print(f"QUESTION (Template: {cand['template']}): {question}")
        print("--------------------------------------------------")
        for i, opt in enumerate(options):
            print(f"  {chr(65+i)}) {opt}")
            
        print("\n[Processing...] Running ML Verifier (Logistic Regression)...")
        best_score = -np.inf
        best_idx = 0
        
        for i, opt in enumerate(options):
            features = extract_features_for_row(article, question, opt, tfidf)
            score = verifier.predict_proba(features)[0, 1]
            if score > best_score:
                best_score = score
                best_idx = i
                
        predicted_letter = chr(65 + best_idx)
        print(f"\n=> ML Verifier Selected: Option {predicted_letter} (Confidence: {best_score:.4f})")
        if predicted_letter == correct_letter:
            print(f"✅ CORRECT! True heuristic answer was {correct_letter} ({correct_answer}).")
        else:
            print(f"❌ INCORRECT. True heuristic answer was {correct_letter} ({correct_answer}).")
            
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    run_app()
