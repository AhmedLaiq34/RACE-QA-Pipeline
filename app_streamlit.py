"""Streamlit Interactive UI for RACE QA System.

This application provides a web-based interface for the RACE QA system, enabling users to:
- Input articles or load random RACE articles
- Generate questions with multiple-choice options
- Submit answers and receive verification from Model A
- Access graduated hints from Model B
- View developer analytics and performance metrics
"""

import random
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from pipeline.evaluate import compute_generation_metrics
from pipeline.inference import (
    generate_distractors,
    generate_question,
    generate_question_auto,  # New automatic function
    get_hints,
    predict_answer,
)


# Configure Streamlit page
st.set_page_config(
    page_title="RACE QA Interactive System",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================================
# Terminal CLI Design System
# ============================================================================

def apply_terminal_theme():
    # Rubric: UX & Error Handling (2 Marks): All four screens usable without reading a manual; friendly error messages for empty input and model failure; loading indicators during inference; sufficient colour contrast (WCAG AA) and readable font sizes; keyboard navigation possible. - 1
    """Apply terminal/phosphor-monitor aesthetic via CSS injection."""
    st.markdown("""
    <style>
    /* Import JetBrains Mono font */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
    
    /* Global styles */
    * {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    /* Main background */
    .stApp {
        background-color: #0a0a0a;
        color: #33ff00;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #33ff00 !important;
        text-transform: uppercase;
        text-shadow: 0 0 5px rgba(51, 255, 0, 0.5);
        letter-spacing: 2px;
    }
    
    /* Text elements */
    p, div, span, label {
        color: #33ff00 !important;
    }
    
    /* Links */
    a {
        color: #ffb000 !important;
        text-decoration: none;
    }
    
    a:hover {
        text-shadow: 0 0 5px rgba(255, 176, 0, 0.5);
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #0a0a0a;
        color: #33ff00;
        border: 1px solid #1f521f;
        border-radius: 0px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        text-transform: uppercase;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background-color: #33ff00;
        color: #0a0a0a;
        border-color: #33ff00;
    }
    
    .stButton > button:active {
        background-color: #1f521f;
    }
    
    /* Primary button */
    .stButton > button[kind="primary"] {
        border: 2px solid #33ff00;
        box-shadow: 0 0 10px rgba(51, 255, 0, 0.3);
    }
    
    /* Text input and text area */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: #0a0a0a;
        color: #33ff00;
        border: 1px solid #1f521f;
        border-radius: 0px;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #33ff00;
        box-shadow: 0 0 5px rgba(51, 255, 0, 0.3);
    }
    
    /* Radio buttons */
    .stRadio > div {
        background-color: #0a0a0a;
        border: 1px dashed #1f521f;
        border-radius: 0px;
        padding: 1rem;
    }
    
    .stRadio > div > label {
        color: #33ff00 !important;
    }
    
    .stRadio > div > div > label {
        color: #33ff00 !important;
        padding: 0.5rem;
        border: 1px solid transparent;
        transition: all 0.2s;
    }
    
    .stRadio > div > div > label:hover {
        border-color: #1f521f;
        background-color: #0f0f0f;
    }
    
    .stRadio > div > div > label[data-checked="true"] {
        background-color: #33ff00;
        color: #0a0a0a !important;
        border-color: #33ff00;
    }
    
    .stRadio > div > div > label[data-checked="true"] > div {
        color: #0a0a0a !important;
    }
    
    /* Divider */
    hr {
        border: none;
        border-top: 1px dashed #1f521f;
        margin: 2rem 0;
    }
    
    /* Info/Success/Warning/Error boxes */
    .stAlert {
        background-color: #0a0a0a;
        border-radius: 0px;
        border-left: 3px solid;
        padding: 1rem;
    }
    
    .stSuccess {
        border-left-color: #33ff00;
        color: #33ff00 !important;
    }
    
    .stSuccess::before {
        content: "[ OK ] ";
        font-weight: bold;
    }
    
    .stError {
        border-left-color: #ff3333;
        color: #ff3333 !important;
    }
    
    .stError::before {
        content: "[ ERR ] ";
        font-weight: bold;
    }
    
    .stWarning {
        border-left-color: #ffb000;
        color: #ffb000 !important;
    }
    
    .stWarning::before {
        content: "[ WARN ] ";
        font-weight: bold;
    }
    
    .stInfo {
        border-left-color: #33ff00;
        color: #33ff00 !important;
    }
    
    .stInfo::before {
        content: "[ INFO ] ";
        font-weight: bold;
    }
    
    /* Metrics */
    .stMetric {
        background-color: #0a0a0a;
        border: 1px solid #1f521f;
        border-radius: 0px;
        padding: 1rem;
    }
    
    .stMetric > div {
        color: #33ff00 !important;
    }
    
    .stMetric label {
        color: #ffb000 !important;
        text-transform: uppercase;
    }
    
    /* Dataframe */
    .stDataFrame {
        border: 1px solid #1f521f;
        border-radius: 0px;
    }
    
    .stDataFrame table {
        background-color: #0a0a0a;
        color: #33ff00;
    }
    
    .stDataFrame th {
        background-color: #1f521f;
        color: #33ff00;
        text-transform: uppercase;
        border: 1px solid #33ff00;
    }
    
    .stDataFrame td {
        border: 1px solid #1f521f;
    }
    
    /* Sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {
        background-color: #0a0a0a;
        border-right: 1px solid #1f521f;
    }
    
    /* Caption text */
    .stCaption {
        color: #1f521f !important;
        font-size: 0.8rem;
    }
    
    /* Download button */
    .stDownloadButton > button {
        background-color: #0a0a0a;
        color: #ffb000;
        border: 1px solid #ffb000;
        border-radius: 0px;
    }
    
    .stDownloadButton > button:hover {
        background-color: #ffb000;
        color: #0a0a0a;
    }
    
    /* Blinking cursor animation */
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0; }
    }
    
    .blinking-cursor::after {
        content: "█";
        animation: blink 1s step-end infinite;
    }
    
    /* CRT scanline effect */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: repeating-linear-gradient(
            0deg,
            rgba(0, 0, 0, 0.15),
            rgba(0, 0, 0, 0.15) 1px,
            transparent 1px,
            transparent 2px
        );
        pointer-events: none;
        z-index: 1000;
    }
    
    /* Container styling */
    .element-container {
        border-radius: 0px;
    }
    
    /* Markdown code blocks */
    code {
        background-color: #0f0f0f;
        color: #33ff00;
        border: 1px solid #1f521f;
        border-radius: 0px;
        padding: 0.2rem 0.4rem;
    }
    
    pre {
        background-color: #0f0f0f;
        border: 1px solid #1f521f;
        border-radius: 0px;
        padding: 1rem;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: #33ff00 transparent transparent transparent;
    }
    </style>
    """, unsafe_allow_html=True)


# Apply theme on load
apply_terminal_theme()


# ============================================================================
# Session State Management
# ============================================================================


def initialize_session_state():
    """Initialize session state variables on first load.
    
    Creates all required session state keys with default values:
    - session_log: List of session entries with user interactions
    - metrics_history: List of generation metrics (BLEU, ROUGE, METEOR)
    - latency_history: List of inference latency measurements
    - Article and question state variables
    - User interaction state flags
    """
    # Analytics data
    if "session_log" not in st.session_state:
        st.session_state.session_log = []
    if "metrics_history" not in st.session_state:
        st.session_state.metrics_history = []
    if "latency_history" not in st.session_state:
        st.session_state.latency_history = []
    
    # Article and input state
    if "article_text_input" not in st.session_state:
        st.session_state.article_text_input = ""
    if "reference_question" not in st.session_state:
        st.session_state.reference_question = None
    if "reference_answer" not in st.session_state:
        st.session_state.reference_answer = None
    if "reference_distractors" not in st.session_state:
        st.session_state.reference_distractors = []
    
    # Current question state
    if "current_article" not in st.session_state:
        st.session_state.current_article = None
    if "current_question" not in st.session_state:
        st.session_state.current_question = None
    if "current_answer" not in st.session_state:
        st.session_state.current_answer = None
    if "current_options" not in st.session_state:
        st.session_state.current_options = []
    if "correct_letter" not in st.session_state:
        st.session_state.correct_letter = None
    if "hints" not in st.session_state:
        st.session_state.hints = []
    if "question_template" not in st.session_state:
        st.session_state.question_template = None
    
    # User interaction state
    if "selected_option" not in st.session_state:
        st.session_state.selected_option = None
    if "answer_submitted" not in st.session_state:
        st.session_state.answer_submitted = False
    if "answer_revealed" not in st.session_state:
        st.session_state.answer_revealed = False
    if "hint_1_revealed" not in st.session_state:
        st.session_state.hint_1_revealed = False
    if "hint_2_revealed" not in st.session_state:
        st.session_state.hint_2_revealed = False
    if "hint_3_revealed" not in st.session_state:
        st.session_state.hint_3_revealed = False



def reset_session():
    """Clear all session state data and reinitialize.
    
    Removes all session state keys and reinitializes with default values.
    This allows users to start fresh without restarting the application.
    """
    # Clear all session state keys
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    # Reinitialize with empty values
    initialize_session_state()



# ============================================================================
# Article Input Component
# ============================================================================


def load_random_article() -> Tuple[str, Optional[str], Optional[str]]:
    """Load a random article from RACE dataset with reference Q&A.
    
    Attempts to load from multiple dataset locations in order:
    1. data/processed/val_exp.csv
    2. data/processed/val_verification.csv
    3. data/raw/val.csv
    
    Returns:
        Tuple of (article_text, reference_question, reference_answer, reference_distractors)
        reference_distractors is a list of 3 distractor strings (excluding the correct answer)
        
    Raises:
        FileNotFoundError: If no valid dataset file is found
    """
    import os
    
    candidates = [
        "data/processed/val_exp.csv",
        "data/processed/val_verification.csv",
        "data/raw/val.csv",
    ]
    
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)
            
            # Determine article column name
            article_col = "article_raw" if "article_raw" in df.columns else "article"
            
            # Get random article
            article = random.choice(
                df[article_col].dropna().astype(str).unique().tolist()
            )
            
            reference_question = None
            reference_answer = None
            reference_distractors = []
            
            # Try to extract reference question and answer
            if "question_raw" in df.columns:
                rows = df[df[article_col] == article]
                if not rows.empty:
                    row = rows.iloc[0]
                    reference_question = str(row["question_raw"])
                    answer_letter = str(row.get("answer", "")).strip()
                    raw_col = f"{answer_letter}_raw"
                    if raw_col in row:
                        reference_answer = str(row[raw_col])
                    
                    # Extract reference distractors (all options except the correct answer)
                    for letter in ["A", "B", "C", "D"]:
                        if letter != answer_letter:
                            distractor_col = f"{letter}_raw"
                            if distractor_col in row:
                                reference_distractors.append(str(row[distractor_col]))
                                
            elif {"question", "answer", "A", "B", "C", "D"}.issubset(df.columns):
                rows = df[df[article_col] == article]
                if not rows.empty:
                    row = rows.iloc[0]
                    reference_question = str(row["question"])
                    answer_letter = str(row["answer"]).strip()
                    if answer_letter in row:
                        reference_answer = str(row[answer_letter])
                    
                    # Extract reference distractors
                    for letter in ["A", "B", "C", "D"]:
                        if letter != answer_letter and letter in row:
                            reference_distractors.append(str(row[letter]))
            
            return article, reference_question, reference_answer, reference_distractors
    
    raise FileNotFoundError(
        "No validation split found in data/processed or data/raw. "
        "Please ensure RACE dataset files exist."
    )



def render_article_input() -> str:
    # Rubric: Screen 1 — Article Input (3 Marks): Text area for pasting or uploading a reading passage; option to load a random RACE dataset sample for quick testing; 'Submit' button triggers both Model A and Model B inference simultaneously; loading indicator shown during inference. - 1
    """Render article input section with text area and random article loader.
    
    Displays:
    - Text area for custom article input
    - Button to load random RACE article
    
    Returns:
        Article text
    """
    st.markdown("### +--- ARTICLE INPUT ---+")
    
    # Initialize the session state key if it doesn't exist
    if "article_text_input" not in st.session_state:
        st.session_state.article_text_input = ""
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        if st.button("[ LOAD RACE ARTICLE ]", use_container_width=True):
            try:
                article_text, ref_question, ref_answer, ref_distractors = load_random_article()
                # Store loaded data in session state
                # Update the text area widget's session state directly
                st.session_state.article_text_input = article_text
                st.session_state.reference_question = ref_question
                st.session_state.reference_answer = ref_answer
                st.session_state.reference_distractors = ref_distractors
                st.success(f"LOADED {len(article_text)} CHARS")
            except FileNotFoundError as e:
                st.error(f"{str(e)}")
            except Exception as e:
                st.error(f"LOAD FAILED: {str(e)}")
    
    with col1:
        # Use the text area with a key - Streamlit will manage the value automatically
        article = st.text_area(
            "$ PASTE ARTICLE OR LOAD FROM DATASET:",
            height=200,
            help="Enter article text or load a random RACE article",
            key="article_text_input"
        )
    
    return article



# ============================================================================
# Question Generation Component
# ============================================================================


def validate_inputs(article: str) -> Tuple[bool, Optional[str]]:
    """Validate article input before question generation.
    
    Args:
        article: Article text
        
    Returns:
        Tuple of (is_valid, error_message)
        If valid, error_message is None
    """
    if len(article.split()) < 10:
        return False, "Article is too short. Please provide at least 10 words."
    
    return True, None



def generate_and_display_question(article: str) -> bool:
    """Generate question from article and store in session state.
    
    For RACE articles: Uses the reference answer from the dataset
    For custom articles: Automatically extracts answer from article
    
    Orchestrates the complete question generation workflow:
    1. Validates inputs
    2. Determines if RACE article (use reference answer) or custom (auto-extract)
    3. Generates question
    4. Generates distractors
    5. Randomizes options
    6. Generates hints
    7. Computes metrics if reference exists
    8. Stores all data in session state
    
    Args:
        article: Article text
        
    Returns:
        True if successful, False otherwise
    """
    # Validate inputs
    is_valid, error_msg = validate_inputs(article)
    if not is_valid:
        st.error(f"❌ {error_msg}")
        return False
    
    with st.spinner("🔄 Generating question..."):
        start_time = time.time()
        
        try:
            # Check if we have a RACE article with reference answer
            reference_answer = st.session_state.get("reference_answer", None)
            
            if reference_answer:
                # RACE article - use reference answer
                # Generate question using reference answer
                candidates = generate_question(article, reference_answer)
                
                if not candidates:
                    st.error("❌ Unable to generate question for this article and answer")
                    return False
                
                # Select first candidate
                selected = candidates[0]
                question = selected["question"]
                answer = reference_answer
                
                # Generate distractors
                distractors = generate_distractors(article, question, answer, n=3)
                template = selected["template"]
                
            else:
                # Custom article - automatically extract answer and generate question
                result = generate_question_auto(article)
                
                if not result:
                    st.error("❌ Unable to generate question from this article. Try a different article with clearer facts.")
                    return False
                
                question = result['question']
                answer = result['answer']
                distractors = result['distractors']
                template = result.get('template', 'auto')
            
            # Combine and randomize options
            options = [answer] + distractors
            random.shuffle(options)
            correct_letter = chr(65 + options.index(answer))
            
            # Generate hints
            hints = get_hints(article, question, n=3)
            
            # Record latency
            latency_ms = (time.time() - start_time) * 1000
            st.session_state.latency_history.append({
                "operation": "question_generation",
                "latency_ms": latency_ms
            })
            
            # Store in session state
            st.session_state.current_article = article
            st.session_state.current_question = question
            st.session_state.current_answer = answer
            st.session_state.current_options = options
            st.session_state.correct_letter = correct_letter
            st.session_state.hints = hints
            st.session_state.question_template = template
            
            # Reset interaction state for new question
            st.session_state.answer_submitted = False
            st.session_state.answer_revealed = False
            st.session_state.hint_1_revealed = False
            st.session_state.hint_2_revealed = False
            st.session_state.hint_3_revealed = False
            
            # Compute metrics if reference exists
            if st.session_state.reference_question:
                metrics = compute_generation_metrics(
                    [question],
                    [st.session_state.reference_question]
                )
                st.session_state.metrics_history.append({
                    "type": "question",
                    "metrics": metrics
                })
            
            # Compute distractor metrics if reference distractors exist
            if st.session_state.get("reference_distractors") and len(st.session_state.reference_distractors) >= 3:
                # Compute metrics for each generated distractor against reference distractors
                # We'll compute average metrics across all distractors
                distractor_metrics_list = []
                for generated_distractor in distractors:
                    # Compare this generated distractor against all reference distractors
                    # and take the best match (highest score)
                    best_metrics = None
                    best_score = -1
                    for ref_distractor in st.session_state.reference_distractors:
                        metrics = compute_generation_metrics(
                            [generated_distractor],
                            [ref_distractor]
                        )
                        # Use BLEU as the primary score for finding best match
                        score = metrics["bleu"]
                        if score > best_score:
                            best_score = score
                            best_metrics = metrics
                    if best_metrics:
                        distractor_metrics_list.append(best_metrics)
                
                # Average the metrics across all distractors
                if distractor_metrics_list:
                    avg_distractor_metrics = {
                        "bleu": np.mean([m["bleu"] for m in distractor_metrics_list]),
                        "rouge_1": np.mean([m["rouge_1"] for m in distractor_metrics_list]),
                        "rouge_l": np.mean([m["rouge_l"] for m in distractor_metrics_list]),
                        "meteor": np.mean([m["meteor"] for m in distractor_metrics_list])
                    }
                    st.session_state.metrics_history.append({
                        "type": "distractor",
                        "metrics": avg_distractor_metrics
                    })
            
            # Compute hint metrics by comparing against answer-containing sentences
            if hints and answer:
                # Extract sentences from article that contain the answer
                import re
                sentences = [s.strip() for s in re.split(r"[.!?]", article) if len(s.strip()) > 10]
                answer_lower = answer.lower()
                reference_sentences = [s for s in sentences if answer_lower in s.lower()]
                
                if reference_sentences:
                    # Compare each hint against the best matching reference sentence
                    hint_metrics_list = []
                    for hint in hints:
                        best_metrics = None
                        best_score = -1
                        for ref_sentence in reference_sentences:
                            metrics = compute_generation_metrics([hint], [ref_sentence])
                            score = metrics["bleu"]
                            if score > best_score:
                                best_score = score
                                best_metrics = metrics
                        if best_metrics:
                            hint_metrics_list.append(best_metrics)
                    
                    # Average the metrics across all hints
                    if hint_metrics_list:
                        avg_hint_metrics = {
                            "bleu": np.mean([m["bleu"] for m in hint_metrics_list]),
                            "rouge_1": np.mean([m["rouge_1"] for m in hint_metrics_list]),
                            "rouge_l": np.mean([m["rouge_l"] for m in hint_metrics_list]),
                            "meteor": np.mean([m["meteor"] for m in hint_metrics_list])
                        }
                        st.session_state.metrics_history.append({
                            "type": "hint",
                            "metrics": avg_hint_metrics
                        })
            
            st.success(f"QUESTION GENERATED SUCCESSFULLY")
            return True
            
        except Exception as e:
            st.error(f"❌ Question generation failed: {str(e)}")
            return False



# ============================================================================
# Question Display and Answer Verification Component
# ============================================================================


def render_question_display():
    # Rubric: Screen 2 — Quiz View (4 Marks): Generated question displayed on screen; 4 options (A–D); Check button; colour-coded correct/incorrect (proper and functional). - 1
    """Render question and multiple-choice options with submission handling.
    
    Displays:
    - Question text with template information
    - Radio buttons for four options (A, B, C, D)
    - Submit button
    - Verification results after submission
    """
    if "current_question" not in st.session_state or not st.session_state.current_question:
        st.info("AWAITING QUESTION GENERATION...")
        return
    
    st.markdown("### +--- QUESTION ---+")
    st.markdown(f"**> {st.session_state.current_question}**")
    st.caption(f"TEMPLATE: {st.session_state.question_template}")
    
    st.markdown("---")
    
    # Radio buttons for options
    options = st.session_state.current_options
    option_labels = [f"[{chr(65+i)}] {opt}" for i, opt in enumerate(options)]
    
    selected_index = st.radio(
        "$ SELECT ANSWER:",
        range(len(options)),
        format_func=lambda i: option_labels[i],
        key="selected_option_radio",
        disabled=st.session_state.get("answer_submitted", False)
    )
    
    # Submit button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(
            "[ SUBMIT ANSWER ]",
            disabled=st.session_state.get("answer_submitted", False),
            use_container_width=True,
            type="primary"
        ):
            handle_answer_submission(selected_index)



def handle_answer_submission(selected_index: int):
    """Process answer submission and display verification results.
    
    Args:
        selected_index: Index of the selected option (0-3)
    """
    if selected_index is None:
        st.warning("⚠️ Please select an answer")
        return
    
    with st.spinner("🔄 Verifying answer..."):
        start_time = time.time()
        
        try:
            # Get data from session state
            article = st.session_state.current_article  # Use stored article
            question = st.session_state.current_question
            options = st.session_state.current_options
            
            # Get model prediction
            predicted_letter = predict_answer(article, question, options)
            user_letter = chr(65 + selected_index)
            correct_letter = st.session_state.correct_letter
            
            # Record latency
            latency_ms = (time.time() - start_time) * 1000
            st.session_state.latency_history.append({
                "operation": "answer_prediction",
                "latency_ms": latency_ms
            })
            
            # Log session entry
            st.session_state.session_log.append({
                "timestamp": datetime.now().isoformat(),
                "article_length": len(article.split()),
                "question": question,
                "user_answer": user_letter,
                "model_prediction": predicted_letter,
                "correct_answer": correct_letter,
                "correct_match": user_letter == predicted_letter,
                "inference_time_ms": latency_ms
            })
            
            # Display result
            st.markdown("---")
            if user_letter == predicted_letter:
                st.success(f"MATCH: MODEL A PREDICTED {predicted_letter}")
            else:
                st.error(f"MISMATCH: MODEL A PREDICTED {predicted_letter}")
            
            # Show correct answer
            correct_text = options[ord(correct_letter) - 65]
            st.info(f"CORRECT ANSWER: [{correct_letter}] {correct_text}")
            
            # Mark as submitted
            st.session_state.answer_submitted = True
            
        except Exception as e:
            st.error(f"❌ Answer prediction failed: {str(e)}")



# ============================================================================
# Hints Panel Component
# ============================================================================


def render_hints_panel():
    # Rubric: Screen 3 — Hint Panel (4 Marks): Collapsible or tabbed panel with three graduated hints from Model B; hints revealed progressively; 'Reveal Answer' button appears only after all hints have been viewed; UI prevents skipping hints. - 1
    """Render hints panel with graduated reveal and answer disclosure.
    
    Displays:
    - Three hint buttons that reveal hints one at a time
    - Reveal Answer button (disables submission)
    - Hint text in info boxes
    - Correct answer in warning box when revealed
    """
    if "hints" not in st.session_state or not st.session_state.hints:
        return
    
    st.markdown("### +--- HINTS ---+")
    st.caption("GRADUATED DISCLOSURE SYSTEM")
    
    hints = st.session_state.hints
    
    # Hint buttons and displays
    for i, hint in enumerate(hints, start=1):
        if st.button(f"[ HINT {i} ]", key=f"hint_btn_{i}", use_container_width=True):
            st.session_state[f"hint_{i}_revealed"] = True
        
        if st.session_state.get(f"hint_{i}_revealed", False):
            st.warning(f"HINT {i}: {hint}")
    
    st.markdown("---")
    
    # Reveal answer button
    if st.button("[ REVEAL ANSWER ]", use_container_width=True, type="secondary"):
        st.session_state.answer_revealed = True
        st.session_state.answer_submitted = True  # Disable submission
    
    if st.session_state.get("answer_revealed", False):
        correct_letter = st.session_state.correct_letter
        options = st.session_state.current_options
        correct_text = options[ord(correct_letter) - 65]
        st.error(f"ANSWER: [{correct_letter}] {correct_text}")



# ============================================================================
# Main Page Layout
# ============================================================================


def render_main_page():
    """Render the main interface page with all components.
    
    Layout:
    - Title
    - Article input section
    - Generate question button
      * RACE articles: Uses reference answer from dataset
      * Custom articles: Automatically extracts answer
    - Two-column layout: Question display | Hints panel
    - Reset session button in sidebar
    """
    st.markdown("# ╔═══════════════════════════════════════════════════════════╗")
    st.markdown("# ║  RACE QA INTERACTIVE SYSTEM — TERMINAL INTERFACE  ║")
    st.markdown("# ╚═══════════════════════════════════════════════════════════╝")
    st.caption("AI-POWERED READING COMPREHENSION VERIFICATION ENGINE")
    
    st.markdown("---")
    
    # Article input section
    article = render_article_input()
    
    st.markdown("---")
    
    # Generate question button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("[ GENERATE QUESTION ]", use_container_width=True, type="primary"):
            generate_and_display_question(article)
    
    st.markdown("---")
    
    # Two-column layout for question and hints
    if st.session_state.get("current_question"):
        col_question, col_hints = st.columns([2, 1])
        
        with col_question:
            render_question_display()
        
        with col_hints:
            render_hints_panel()
    
    # Reset session button in sidebar
    with st.sidebar:
        st.markdown("### SYSTEM CONTROLS")
        st.markdown("---")
        if st.button("[ RESET SESSION ]", use_container_width=True):
            reset_session()
            st.rerun()



# ============================================================================
# Developer Dashboard Components
# ============================================================================


def render_model_metrics(metric_type: str):
    # Rubric: Evaluation (BLEU/ROUGE/METEOR) (3 Marks): BLEU, ROUGE, METEOR scores reported for distractor generation quality. - 1
    """Display average metrics for question or distractor generation.
    
    Args:
        metric_type: "question" or "distractor"
    """
    metrics_history = st.session_state.metrics_history
    filtered = [m for m in metrics_history if m["type"] == metric_type]
    
    if not filtered:
        st.info(f"NO {metric_type.upper()} METRICS AVAILABLE. GENERATE QUESTIONS TO POPULATE.")
        return
    
    # Take last 10 entries
    recent = filtered[-10:]
    
    # Compute averages
    avg_bleu = np.mean([m["metrics"]["bleu"] for m in recent])
    avg_rouge1 = np.mean([m["metrics"]["rouge_1"] for m in recent])
    avg_rougel = np.mean([m["metrics"]["rouge_l"] for m in recent])
    avg_meteor = np.mean([m["metrics"]["meteor"] for m in recent])
    
    # Display in columns
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("BLEU", f"{avg_bleu:.4f}")
    col2.metric("ROUGE-1", f"{avg_rouge1:.4f}")
    col3.metric("ROUGE-L", f"{avg_rougel:.4f}")
    col4.metric("METEOR", f"{avg_meteor:.4f}")
    
    st.caption(f"BASED ON LAST {len(recent)} INFERENCE(S)")



def render_latency_metrics():
    """Display average latency by operation type."""
    latency_history = st.session_state.latency_history
    
    if not latency_history:
        st.info("NO LATENCY DATA AVAILABLE. GENERATE QUESTIONS TO POPULATE.")
        return
    
    # Group by operation
    operations = {}
    for entry in latency_history:
        op = entry["operation"]
        if op not in operations:
            operations[op] = []
        operations[op].append(entry["latency_ms"])
    
    # Display averages in columns
    cols = st.columns(len(operations))
    for idx, (op, latencies) in enumerate(operations.items()):
        avg_latency = np.mean(latencies)
        cols[idx].metric(
            op.replace("_", " ").upper(),
            f"{avg_latency:.2f} MS"
        )



def render_session_log():
    """Display session log table with export functionality."""
    session_log = st.session_state.session_log
    
    if not session_log:
        st.info("NO SESSION ENTRIES. ANSWER QUESTIONS TO POPULATE LOG.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(session_log)
    
    # Display table
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Export button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("[ EXPORT TO CSV ]", use_container_width=True):
            csv = df.to_csv(index=False)
            st.download_button(
                label="[ DOWNLOAD CSV ]",
                data=csv,
                file_name=f"session_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )



def render_developer_dashboard():
    # Rubric: Screen 4 — Analytics Dashboard (2 Marks): Model metrics displayed; inference latency shown; CSV export available. - 1
    """Render developer analytics dashboard with metrics and logs."""
    st.markdown("# ╔═══════════════════════════════════════════════════════════╗")
    st.markdown("# ║  DEVELOPER DASHBOARD — SYSTEM ANALYTICS  ║")
    st.markdown("# ╚═══════════════════════════════════════════════════════════╝")
    st.caption("PERFORMANCE METRICS AND SESSION ANALYTICS")
    
    st.markdown("---")
    
    # Model A Performance Metrics
    st.markdown("### +--- MODEL A: QUESTION GENERATION ---+")
    st.caption("QUALITY METRICS (COMPARED TO REFERENCE QUESTIONS)")
    render_model_metrics("question")
    
    st.markdown("---")
    
    # Model B Performance Metrics
    st.markdown("### +--- MODEL B: DISTRACTOR GENERATION ---+")
    st.caption("QUALITY METRICS (COMPARED TO REFERENCE DISTRACTORS)")
    render_model_metrics("distractor")
    
    st.markdown("---")
    
    # Hint Generation Metrics
    st.markdown("### +--- HINT GENERATOR: SENTENCE SELECTION ---+")
    st.caption("QUALITY METRICS (COMPARED TO ANSWER-CONTAINING SENTENCES)")
    render_model_metrics("hint")
    
    st.markdown("---")
    
    # Latency Tracking
    st.markdown("### +--- LATENCY TRACKING ---+")
    st.caption("AVERAGE INFERENCE TIME PER OPERATION")
    render_latency_metrics()
    
    st.markdown("---")
    
    # Session Log
    st.markdown("### +--- SESSION LOG ---+")
    st.caption("ALL USER INTERACTIONS AND MODEL PREDICTIONS")
    render_session_log()



# ============================================================================
# Main Application Entry Point
# ============================================================================


def main():
    """Main application entry point with page routing."""
    # Initialize session state
    initialize_session_state()
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("### NAVIGATION")
        page = st.radio(
            "SELECT PAGE:",
            ["MAIN INTERFACE", "DEVELOPER DASHBOARD"],
            label_visibility="collapsed"
        )
    
    # Route to appropriate page
    if page == "MAIN INTERFACE":
        render_main_page()
    else:
        render_developer_dashboard()


if __name__ == "__main__":
    main()
