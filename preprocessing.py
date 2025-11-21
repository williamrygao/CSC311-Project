# preprocessing.py
import re
import numpy as np
import pandas as pd

NUM_COLS = [
    "How likely are you to use this model for academic tasks?",
    "Based on your experience, how often has this model given you a response that felt suboptimal?",
    "How often do you expect this model to provide responses with references or supporting evidence?",
    "How often do you verify this model's responses?",
]

BEST_TASKS = [
    "Brainstorming or generating creative ideas",
    "Converting content between formats (e.g.",
    "Data processing or analysis",
    "Drafting professional text (e.g.",
    "Explaining complex concepts simply",
    "LaTeX)",
    "Math computations",
    "Writing or debugging code",
    "Writing or editing essays/reports",
    "emails",
    "résumés)",
]

SUBOPT_TASKS = BEST_TASKS[:]  # same set

BEST_COL = "Which types of tasks do you feel this model handles best? (Select all that apply.)"
SUBOPT_COL = "For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)"

def extract_rating(response):
    """Extract leading integer from strings like '3 — Neutral / Unsure'."""
    m = re.match(r"^(\d+)", str(response))
    return int(m.group(1)) if m else np.nan

def build_features(df: pd.DataFrame) -> np.ndarray:
    """
    Create feature matrix X from the input DataFrame.
    - Numeric features: impute missing with 3
    - Multi-select features: 11 binary indicators for best/suboptimal tasks
    Returns:
        X: np.ndarray of shape (n_samples, 26)
    """
    # Numeric
    num_feats = []
    for col in NUM_COLS:
        col_vals = df[col].apply(extract_rating).astype(float).fillna(3.0)
        num_feats.append(col_vals.values.reshape(-1, 1))
    num_mat = np.hstack(num_feats)  # (n, 4)

    # Multi-select
    n = len(df)
    best_mat = np.zeros((n, len(BEST_TASKS)), dtype=float)
    subopt_mat = np.zeros((n, len(SUBOPT_TASKS)), dtype=float)

    for i, (_, row) in enumerate(df.iterrows()):
        best_resp = str(row.get(BEST_COL, ""))
        subopt_resp = str(row.get(SUBOPT_COL, ""))

        for j, task in enumerate(BEST_TASKS):
            if task in best_resp:
                best_mat[i, j] = 1.0
        for j, task in enumerate(SUBOPT_TASKS):
            if task in subopt_resp:
                subopt_mat[i, j] = 1.0

    X = np.hstack([num_mat, best_mat, subopt_mat])
    return X

def load_and_preprocess(filename: str) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Load CSV, return features, labels, and student groups.
    Expects columns: 'student_id', 'label', plus survey columns.
    """
    df = pd.read_csv(filename)
    X = build_features(df)
    label_to_index = {"ChatGPT": 0, "Claude": 1, "Gemini": 2}
    y = df["label"].map(label_to_index).values
    groups = df["student_id"].values
    return X, y, groups
