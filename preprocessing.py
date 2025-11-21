# preprocessing_no_text.py
import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

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

def extract_rating(response) -> float:
    m = re.match(r"^(\d+)", str(response))
    return float(m.group(1)) if m else np.nan

class PreprocessorNoText:
    def __init__(self, degree=2, interaction_only=True):
        self.scaler = StandardScaler()
        self.poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)

    def fit_transform(self, df: pd.DataFrame):
        X = self._base_features(df)
        X_scaled = self.scaler.fit_transform(X)
        X_poly = self.poly.fit_transform(X_scaled)
        return X_poly

    def transform(self, df: pd.DataFrame):
        X = self._base_features(df)
        X_scaled = self.scaler.transform(X)
        X_poly = self.poly.transform(X_scaled)
        return X_poly

    def _base_features(self, df: pd.DataFrame):
        # Numeric
        num_mat = np.hstack([
            df[col].apply(extract_rating).fillna(3.0).astype(float).values.reshape(-1, 1)
            for col in NUM_COLS
        ])
        # Multi-select
        n = len(df)
        best_mat = np.zeros((n, len(BEST_TASKS)), dtype=float)
        subopt_mat = np.zeros((n, len(SUBOPT_TASKS)), dtype=float)
        for i, (_, row) in enumerate(df.iterrows()):
            best_resp = str(row.get(BEST_COL, ""))
            subopt_resp = str(row.get(SUBOPT_COL, ""))
            for j, task in enumerate(BEST_TASKS):
                if task in best_resp: best_mat[i,j]=1.0
            for j, task in enumerate(SUBOPT_TASKS):
                if task in subopt_resp: subopt_mat[i,j]=1.0
        X = np.hstack([num_mat, best_mat, subopt_mat])
        return X


def load_and_preprocess_no_text(path):
    df = pd.read_csv(path)
    y = df["label"].map({"ChatGPT":0, "Claude":1, "Gemini":2}).values
    groups = df["student_id"].values
    preproc = PreprocessorNoText()
    X = preproc.fit_transform(df)
    return X, y, groups, preproc
