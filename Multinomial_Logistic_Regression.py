import numpy as np
import pandas as pd
import re

from sklearn.linear_model import LogisticRegression

# Upload training_data_clean.csv to Colab (using the upload UI) and then:
df = pd.read_csv("training_data_clean.csv")

df.head()



# Columns that are 1–5 ratings
NUM_COLS = [
    "How likely are you to use this model for academic tasks?",
    "Based on your experience, how often has this model given you a response that felt suboptimal?",
    "How often do you expect this model to provide responses with references or supporting evidence?",
    "How often do you verify this model's responses?",
]

# Fixed list of tasks we saw in the data (both for "best" and "suboptimal")
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
    """
    Extract leading integer from strings like '3 — Neutral / Unsure'.
    """
    m = re.match(r"^(\d+)", str(response))
    return int(m.group(1)) if m else np.nan


def build_features(df: pd.DataFrame) -> np.ndarray:
    """
    Turn a DataFrame into a feature matrix X of shape (n_samples, n_features).
    - Numeric features: 4 rating questions -> impute missing as 3 (neutral).
    - Multi-select: 11 binary features for BEST_TASKS and 11 for SUBOPT_TASKS.
    """
    # Numeric features
    num_feats = []
    for col in NUM_COLS:
        col_vals = df[col].apply(extract_rating).astype(float)
        col_vals = col_vals.fillna(3.0)  # neutral rating
        num_feats.append(col_vals.values.reshape(-1, 1))
    num_mat = np.hstack(num_feats)  # shape (n, 4)

    # Multi-select features
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

    X = np.hstack([num_mat, best_mat, subopt_mat])  # total 4 + 11 + 11 = 26 features
    return X



# Build features and labels (same as before)
X = build_features(df)
label_to_index = {"ChatGPT": 0, "Claude": 1, "Gemini": 2}
y = df["label"].map(label_to_index).values

# 70/30 split (same as before)
n = len(X)
n_train = int(0.7 * n)
X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

# Logistic regression with tuned regularization
clf = LogisticRegression(
    solver="lbfgs",
    max_iter=1000,
)

clf.fit(X_train, y_train)

print("Training accuracy:", clf.score(X_train, y_train))
print("Test accuracy:", clf.score(X_test, y_test))

# Save weights for pred.py
W = clf.coef_        # (3, n_features)
b = clf.intercept_   # (3,)
np.savez("lr_params.npz", W=W, b=b)



W = clf.coef_        # shape (3, 26)
b = clf.intercept_   # shape (3,)

np.savez("lr_params.npz", W=W, b=b)



"""
pred.py - Multinomial logistic regression predictor for the GenAI survey project.

This script:
  - reads a CSV with the same columns as training_data_clean.csv (except 'label')
  - builds a 26-dimensional feature vector per row
  - loads pre-trained logistic regression parameters from lr_params.npz
  - returns a list of predictions: 'ChatGPT', 'Claude', or 'Gemini'

Allowed imports: standard library, numpy, pandas.
"""

import os
import re
from typing import List

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Constants: feature definitions must match the ones used in training
# ---------------------------------------------------------------------

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

INDEX_TO_LABEL = ["ChatGPT", "Claude", "Gemini"]  # 0 -> ChatGPT, 1 -> Claude, 2 -> Gemini


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def extract_rating(response) -> float:
    """
    Extract leading integer from strings like '3 — Neutral / Unsure'.
    If extraction fails, return NaN.
    """
    m = re.match(r"^(\d+)", str(response))
    return float(m.group(1)) if m else np.nan


def build_features(df: pd.DataFrame) -> np.ndarray:
    """
    Create feature matrix X from the input DataFrame.

    Features:
      - 4 numeric rating questions -> impute missing as 3.0
      - 11 binary indicators for BEST_TASKS
      - 11 binary indicators for SUBOPT_TASKS

    Returns:
      X: numpy array of shape (n_samples, 26)
    """
    # Numeric features
    num_feats = []
    for col in NUM_COLS:
        col_vals = df[col].apply(extract_rating).astype(float)
        # Impute missing with neutral rating 3
        col_vals = col_vals.fillna(3.0)
        num_feats.append(col_vals.values.reshape(-1, 1))
    num_mat = np.hstack(num_feats)  # (n, 4)

    # Multi-select features
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

    X = np.hstack([num_mat, best_mat, subopt_mat])  # (n, 26)
    return X


def load_parameters():
    """
    Load logistic regression parameters from lr_params.npz.

    Works both:
      - inside a normal pred.py file (where __file__ exists)
      - inside a Jupyter/Colab notebook (where __file__ does NOT exist)
    """
    import os

    # If running inside a notebook, __file__ does not exist
    if "__file__" in globals():
        base_dir = os.path.dirname(os.path.abspath(__file__))
    else:
        # In a notebook, assume lr_params.npz is in current working directory
        base_dir = os.getcwd()

    params_path = os.path.join(base_dir, "lr_params.npz")

    data = np.load(params_path)
    W = data["W"]
    b = data["b"]
    return W, b


def predict_scores(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute class scores for each example (no softmax needed for argmax).
    X: (n, d)
    W: (C, d)
    b: (C,)
    Returns: scores (n, C)
    """
    # scores_i = X_i @ W^T + b
    return X @ W.T + b


def predict_labels(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> List[str]:
    """
    Given features X and model parameters W, b, return list of label strings.
    """
    scores = predict_scores(X, W, b)   # (n, 3)
    class_indices = np.argmax(scores, axis=1)  # (n,)
    labels = [INDEX_TO_LABEL[idx] for idx in class_indices]
    return labels


# ---------------------------------------------------------------------
# Public API required by the course: predict_all(filename)
# ---------------------------------------------------------------------

def predict_all(filename: str) -> List[str]:
    """
    Read the CSV file `filename`, build features, and return predictions.

    Parameters:
        filename: path to the test CSV file.

    Returns:
        A list of strings, one of: 'ChatGPT', 'Claude', 'Gemini' for each row.
    """
    # Load data
    df = pd.read_csv(filename)

    # Build features
    X = build_features(df)

    # Load trained parameters
    W, b = load_parameters()

    # Predict labels
    predictions = predict_labels(X, W, b)
    return predictions



# Use part of training data (without the label column) as a fake "test" file
df_no_label = df.drop(columns=["label"])
df_no_label.to_csv("fake_test.csv", index=False)

preds = predict_all("fake_test.csv")
print(len(preds), preds[:10])
