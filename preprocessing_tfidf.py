import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

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

SUBOPT_TASKS = BEST_TASKS[:]

BEST_COL = "Which types of tasks do you feel this model handles best? (Select all that apply.)"
SUBOPT_COL = "For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)"

TEXT_COLS = [
    "In your own words, what kinds of tasks would you use this model for?",
    "Think of one task where this model gave you a suboptimal response. What did the response look like, and why did you find it suboptimal?",
    "When you verify a response from this model, how do you usually go about it?"
]

def extract_rating(response) -> float:
    """Extract leading integer from Likert scale response."""
    m = re.match(r"^(\d+)", str(response))
    return float(m.group(1)) if m else np.nan

class Preprocessor:
    def __init__(self, max_features=300, ngram_range=(1,1)):
        self.scaler = StandardScaler()
        self.vectorizers = {
            col: TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
            for col in TEXT_COLS
        }

    def fit_transform(self, df: pd.DataFrame):
        # ------------------------
        # Numeric features
        # ------------------------
        num_mat = []
        for col in NUM_COLS:
            col_vals = df[col].apply(extract_rating).fillna(3.0).astype(float)
            num_mat.append(col_vals.values.reshape(-1, 1))
        num_mat = np.hstack(num_mat)  # (n_samples, 4)
        num_mat = self.scaler.fit_transform(num_mat)

        # ------------------------
        # Multi-select categorical
        # ------------------------
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

        # ------------------------
        # Text features: TF-IDF
        # ------------------------
        text_mats = []
        for col in TEXT_COLS:
            X_text = self.vectorizers[col].fit_transform(df[col].fillna("")).toarray()
            text_mats.append(X_text)
        text_mat = np.hstack(text_mats) if text_mats else np.zeros((n, 0))

        # ------------------------
        # Concatenate everything
        # ------------------------
        X = np.hstack([num_mat, best_mat, subopt_mat, text_mat])
        return X

    def transform(self, df: pd.DataFrame):
        n = len(df)
        # Numeric
        num_mat = []
        for col in NUM_COLS:
            col_vals = df[col].apply(extract_rating).fillna(3.0).astype(float)
            num_mat.append(col_vals.values.reshape(-1, 1))
        num_mat = np.hstack(num_mat)
        num_mat = self.scaler.transform(num_mat)

        # Multi-select
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

        # Text
        text_mats = []
        for col in TEXT_COLS:
            X_text = self.vectorizers[col].transform(df[col].fillna("")).toarray()
            text_mats.append(X_text)
        text_mat = np.hstack(text_mats) if text_mats else np.zeros((n, 0))

        X = np.hstack([num_mat, best_mat, subopt_mat, text_mat])
        return X
    
def load_and_preprocess_with_preprocessor(path, max_features=300, ngram_range=(1,1)):
    """
    Returns:
        X : np.ndarray
        y : np.ndarray
        groups : np.ndarray
        preproc : Preprocessor (fitted on this data)
    """
    df = pd.read_csv(path)
    
    # Labels
    y = df["label"].map({"ChatGPT":0, "Claude":1, "Gemini":2}).values
    groups = df["student_id"].values
    
    # Fit preprocessor on training data
    preproc = Preprocessor(max_features=max_features, ngram_range=ngram_range)
    X = preproc.fit_transform(df)
    
    return X, y, groups, preproc
