"""Build the reproducible EDA and training notebook."""

import os
import textwrap

import nbformat as nbf


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
NOTEBOOK_PATH = os.path.join(BASE_DIR, "EDA_and_Training.ipynb")
PIPELINE_DIR = os.path.join(PROJECT_ROOT, "pipeline")


def _dedent(source):
    return textwrap.dedent(source).strip()


def _read_pipeline_file(filename):
    with open(os.path.join(PIPELINE_DIR, filename), "r", encoding="utf-8") as handle:
        return handle.read().rstrip()


def _writefile_cell(target_path, source):
    return f"%%writefile {target_path}\n{source}\n"


def build_notebook():
    notebook = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell("# RACE QA Pipeline - EDA and Training"))

    cells.append(nbf.v4.new_markdown_cell("## Section 1 - Environment & Data Loading"))
    cells.append(
        nbf.v4.new_code_cell(
            _dedent(
                """
                import os
                import re
                import string
                import warnings
                from pathlib import Path

                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                import numpy as np
                import pandas as pd

                warnings.filterwarnings('ignore')
                PROJECT_ROOT = Path.cwd().resolve().parent if Path.cwd().name == 'notebooks' else Path.cwd().resolve()
                RAW_DIR = PROJECT_ROOT / 'data' / 'raw'
                PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
                FIG_DIR = PROCESSED_DIR / 'figures'
                RAW_DIR.mkdir(parents=True, exist_ok=True)
                PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
                FIG_DIR.mkdir(parents=True, exist_ok=True)
                print(PROJECT_ROOT)
                """
            )
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            _dedent(
                """
                # Optional download hook. Existing CSVs are used when present.
                if not (RAW_DIR / 'train.csv').exists():
                    import kagglehub
                    print('Download the RACE dataset with kagglehub, then standardize to article, question, A, B, C, D, answer.')
                    print('This project already expects train.csv, val.csv, and test.csv under data/raw/.')

                train_df = pd.read_csv(RAW_DIR / 'train.csv')
                val_df = pd.read_csv(RAW_DIR / 'val.csv')
                test_df = pd.read_csv(RAW_DIR / 'test.csv')
                required = ['article', 'question', 'A', 'B', 'C', 'D', 'answer']
                for name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
                    missing = set(required) - set(df.columns)
                    if missing:
                        raise ValueError(f'{name} is missing columns: {sorted(missing)}')
                    print(name, df.shape)
                """
            )
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## Section 2 - EDA"))
    cells.append(
        nbf.v4.new_code_cell(
            _dedent(
                """
                def save_plot(name):
                    plt.tight_layout()
                    plt.savefig(FIG_DIR / name, dpi=150)
                    plt.close()

                train_df['answer'].value_counts().sort_index().plot(kind='bar', title='Answer distribution')
                save_plot('answer_distribution.png')

                train_df['article'].astype(str).str.split().str.len().plot(kind='hist', bins=50, title='Article length')
                save_plot('article_length_hist.png')

                train_df['question'].astype(str).str.split().str.len().plot(kind='hist', bins=40, title='Question length')
                save_plot('question_length_hist.png')

                option_lengths = train_df[['A', 'B', 'C', 'D']].astype(str).applymap(lambda x: len(x.split()))
                option_lengths.boxplot()
                plt.title('Option length comparison')
                save_plot('option_length_comparison.png')

                qtype = train_df['question'].astype(str).str.extract(r'^(\\w+)', expand=False).str.lower().value_counts().head(15)
                qtype.plot(kind='bar', title='Question type breakdown')
                save_plot('question_type_breakdown.png')

                split_balance = pd.DataFrame({
                    'train': train_df['answer'].value_counts(normalize=True).sort_index(),
                    'val': val_df['answer'].value_counts(normalize=True).sort_index(),
                    'test': test_df['answer'].value_counts(normalize=True).sort_index(),
                })
                split_balance.plot(kind='bar', title='Answer balance across splits')
                save_plot('answer_balance_across_splits.png')

                summary = pd.DataFrame({
                    'article_words': train_df['article'].astype(str).str.split().str.len().describe(),
                    'question_words': train_df['question'].astype(str).str.split().str.len().describe(),
                })
                summary
                """
            )
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## Section 3 - Preprocessing"))
    cells.append(
        nbf.v4.new_code_cell(
            _dedent(
                """
                import sys
                sys.path.insert(0, str(PROJECT_ROOT))
                from pipeline.preprocessing import clean_text, expand_df, prepare_text_columns, preprocess_and_build

                train_exp, val_exp, test_exp = preprocess_and_build()
                print(train_exp.shape, val_exp.shape, test_exp.shape)
                """
            )
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## Section 4 - Model A (Supervised)"))
    cells.append(
        nbf.v4.new_code_cell(
            _dedent(
                """
                from pipeline.model_a_train import main as train_model_a
                train_model_a()
                """
            )
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## Section 5 - Unsupervised & Semi-Supervised"))
    cells.append(
        nbf.v4.new_code_cell(
            _dedent(
                """
                from scipy.sparse import load_npz
                from sklearn.cluster import KMeans
                from sklearn.decomposition import TruncatedSVD
                from sklearn.mixture import GaussianMixture
                from sklearn.semi_supervised import LabelPropagation

                X_train = load_npz(PROCESSED_DIR / 'X_train.npz')
                y_train = np.load(PROCESSED_DIR / 'y_train.npy')
                sample_n = min(5000, X_train.shape[0])
                X_sample = X_train[:sample_n]
                y_sample = y_train[:sample_n]

                inertias = []
                for k in range(2, 8):
                    km = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(X_sample)
                    inertias.append(km.inertia_)
                plt.plot(range(2, 8), inertias, marker='o')
                plt.title('K-Means elbow')
                save_plot('kmeans_elbow.png')

                svd = TruncatedSVD(n_components=2, random_state=42)
                coords = svd.fit_transform(X_sample)
                plt.scatter(coords[:, 0], coords[:, 1], c=y_sample, s=4, alpha=0.5)
                plt.title('SVD projection')
                save_plot('svd_projection.png')

                gmm = GaussianMixture(n_components=2, random_state=42).fit(coords)
                labels = np.full(sample_n, -1)
                labels[:max(1, sample_n // 10)] = y_sample[:max(1, sample_n // 10)]
                lp = LabelPropagation().fit(coords, labels)
                {'gmm_components': gmm.n_components, 'label_prop_classes': sorted(set(lp.transduction_.tolist()))}
                """
            )
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## Section 6 - Question Generation"))
    cells.append(
        nbf.v4.new_code_cell(
            _dedent(
                """
                from pipeline.inference import generate_question

                example = train_df.iloc[0]
                answer = example[example['answer']]
                generate_question(example['article'], answer)[:5]
                """
            )
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## Section 7 - Model B"))
    cells.append(
        nbf.v4.new_code_cell(
            _dedent(
                """
                from pipeline.model_b_train import main as train_model_b
                train_model_b()
                """
            )
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## Section 8 - Final Evaluation"))
    cells.append(
        nbf.v4.new_code_cell(
            _dedent(
                """
                from pipeline.evaluate import compute_generation_metrics, compute_metrics
                from pipeline.inference import generate_distractors, get_hints, predict_answer

                row = test_df.iloc[0]
                options = [row[o] for o in ['A', 'B', 'C', 'D']]
                pred = predict_answer(row['article'], row['question'], options)
                distractors = generate_distractors(row['article'], row['question'], row[row['answer']])
                hints = get_hints(row['article'], row['question'])
                generation_metrics = compute_generation_metrics([' '.join(distractors)], [' '.join([row[o] for o in ['A', 'B', 'C', 'D'] if o != row['answer']])])
                {'predicted_answer': pred, 'distractors': distractors, 'hints': hints, 'generation_metrics': generation_metrics}
                """
            )
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## Section 9 - Export Scripts"))
    cells.append(nbf.v4.new_code_cell("from pathlib import Path\nPath('src').mkdir(exist_ok=True)"))
    for filename in ["preprocessing.py", "evaluate.py", "model_a_train.py", "model_b_train.py", "inference.py"]:
        cells.append(nbf.v4.new_code_cell(_writefile_cell(f"src/{filename}", _read_pipeline_file(filename))))

    notebook["cells"] = cells
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as handle:
        nbf.write(notebook, handle)
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    build_notebook()
