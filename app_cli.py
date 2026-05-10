"""Interactive CLI for the refactored RACE QA pipeline."""

import os
import random

import pandas as pd

from pipeline.evaluate import compute_generation_metrics
from pipeline.inference import generate_distractors, generate_question, get_hints, predict_answer


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = BASE_DIR
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


def _load_random_article():
    candidates = [
        os.path.join(PROCESSED_DIR, "val_exp.csv"),
        os.path.join(PROCESSED_DIR, "val_verification.csv"),
        os.path.join(RAW_DIR, "val.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)
            article_col = "article_raw" if "article_raw" in df.columns else "article"
            article = random.choice(df[article_col].dropna().astype(str).unique().tolist())
            reference_question = None
            reference_answer = None
            if "question_raw" in df.columns:
                rows = df[df[article_col] == article]
                if not rows.empty:
                    reference_question = str(rows.iloc[0]["question_raw"])
                    answer_letter = str(rows.iloc[0].get("answer", "")).strip()
                    raw_col = f"{answer_letter}_raw"
                    if raw_col in rows.columns:
                        reference_answer = str(rows.iloc[0][raw_col])
            elif {"question", "answer", "A", "B", "C", "D"}.issubset(df.columns):
                rows = df[df[article_col] == article]
                if not rows.empty:
                    reference_question = str(rows.iloc[0]["question"])
                    answer_letter = str(rows.iloc[0]["answer"]).strip()
                    if answer_letter in rows.columns:
                        reference_answer = str(rows.iloc[0][answer_letter])
            return article, reference_question, reference_answer
    raise FileNotFoundError("No validation split found in data/processed or data/raw.")


def _prompt_article():
    print("\nPaste your article text below.")
    return input("> ").strip(), None, None


def _prompt_answer(default_answer=None):
    if default_answer:
        print(f"\nReference answer found: {default_answer}")
        use_reference = input("Use this answer for generation? (Y/n): ").strip().lower()
        if use_reference in {"", "y", "yes"}:
            return default_answer
    return input("\nEnter the correct answer phrase to build questions around:\n> ").strip()


def _prepare_question_display(article, candidate):
    question = candidate["question"]
    correct_answer = candidate["answer"]
    distractors = generate_distractors(article, question, correct_answer, n=3)
    options = [correct_answer] + distractors
    random.shuffle(options)
    correct_letter = chr(65 + options.index(correct_answer))
    predicted_letter = predict_answer(article, question, options)
    hints = get_hints(article, question, n=3)
    return {
        "candidate": candidate,
        "options": options,
        "correct_letter": correct_letter,
        "predicted_letter": predicted_letter,
        "hints": hints,
    }


def _display_question(prepared, reference_question=None):
    candidate = prepared["candidate"]
    question = candidate["question"]
    correct_answer = candidate["answer"]

    print("\n" + "-" * 60)
    print(f"QUESTION ({candidate['template']}): {question}")
    print("-" * 60)
    for idx, option in enumerate(prepared["options"]):
        print(f"  {chr(65 + idx)}) {option}")

    print(f"\nVerifier selected: {prepared['predicted_letter']}")
    print(f"Expected answer: {prepared['correct_letter']} ({correct_answer})")
    print("\nHints:")
    for idx, hint in enumerate(prepared["hints"], start=1):
        print(f"  {idx}. {hint}")

    if reference_question:
        metrics = compute_generation_metrics([question], [reference_question])
        print("\nGeneration metrics against reference question:")
        print(
            "  BLEU={bleu:.4f} ROUGE-1={rouge_1:.4f} ROUGE-L={rouge_l:.4f} METEOR={meteor:.4f}".format(
                **metrics
            )
        )


def _build_display_candidates(article, answer, reference_question):
    candidates = []
    seen = set()
    if reference_question:
        candidates.append(
            {
                "question": reference_question,
                "answer": answer,
                "source_sentence": "RACE reference question",
                "template": "reference",
            }
        )
        seen.add(reference_question.strip().lower())

    for candidate in generate_question(article, answer):
        key = candidate["question"].strip().lower()
        if key not in seen:
            seen.add(key)
            candidates.append(candidate)
    return candidates


def run_app():
    print("\n" + "=" * 60)
    print("  RACE QA Generation, Distractors, Hints & Verification")
    print("=" * 60)

    while True:
        print("\nOptions:")
        print("  1. Enter your own article")
        print("  2. Load a random RACE article")
        print("  3. Exit")
        choice = input("Select an option (1/2/3): ").strip()

        if choice == "3":
            break
        if choice == "1":
            article, reference_question, reference_answer = _prompt_article()
        elif choice == "2":
            try:
                article, reference_question, reference_answer = _load_random_article()
                print("\nLoaded random article:")
                print(article[:600] + ("..." if len(article) > 600 else ""))
            except Exception as exc:
                print(f"Error loading dataset: {exc}")
                continue
        else:
            continue

        if len(article.split()) < 4:
            print("Article is too short.")
            continue

        answer = _prompt_answer(reference_answer)
        if not answer:
            print("A correct answer phrase is required for question generation.")
            continue

        print("\nGenerating candidate questions...")
        candidates = _build_display_candidates(article, answer, reference_question)
        if not candidates:
            print("No question candidates could be generated for that answer.")
            continue

        verified = []
        prepared_candidates = []
        for candidate in candidates:
            if len(verified) >= 3:
                break
            prepared = _prepare_question_display(article, candidate)
            prepared_candidates.append(prepared)
            if prepared["predicted_letter"] == prepared["correct_letter"]:
                verified.append(prepared)

        display_candidates = verified or prepared_candidates[:3]
        if verified:
            print(f"Showing {len(display_candidates)} verifier-approved candidate(s).")
        else:
            print("No candidates passed verification; showing generated candidates for inspection.")

        for idx, prepared in enumerate(display_candidates[:3], start=1):
            print(f"\nCandidate {idx}:")
            _display_question(prepared, reference_question)

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    run_app()
