# logistic_regression_tfidf.py
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocessing_tfidf import Preprocessor  # your existing Preprocessor class

# ------------------------
# Load data
# ------------------------
df = pd.read_csv("train.csv")
y = df["label"].map({"ChatGPT": 0, "Claude": 1, "Gemini": 2}).values
groups = df["student_id"].values

# ------------------------
# Optimal hyperparameters
# ------------------------
max_features = 300
C = 5.0
penalty = 'l2'
solver = 'saga'   # note: saga supports L2
max_iter = 5000

# ------------------------
# Initialize preprocessor and transform data
# ------------------------
preproc = Preprocessor(max_features=max_features)
X = preproc.fit_transform(df)

# ------------------------
# Grouped 5-fold cross-validation
# ------------------------
gkf = GroupKFold(n_splits=5)
fold_acc, fold_prec, fold_rec, fold_f1 = [], [], [], []

print("Grouped 5-fold cross-validation (TF-IDF):")
for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    clf = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=max_iter)
    clf.fit(X_train, y_train)
    y_val_pred = clf.predict(X_val)

    acc = accuracy_score(y_val, y_val_pred)
    prec = precision_score(y_val, y_val_pred, average='macro')
    rec = recall_score(y_val, y_val_pred, average='macro')
    f1 = f1_score(y_val, y_val_pred, average='macro')

    fold_acc.append(acc)
    fold_prec.append(prec)
    fold_rec.append(rec)
    fold_f1.append(f1)

    print(f"Fold {fold}: Acc={acc:.3f}, Prec={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}")

print("\nMean CV metrics:")
print(f"Accuracy={np.mean(fold_acc):.3f}, Precision={np.mean(fold_prec):.3f}, "
      f"Recall={np.mean(fold_rec):.3f}, F1={np.mean(fold_f1):.3f}")

# ------------------------
# Train final model on all data
# ------------------------
final_clf = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=max_iter)
final_clf.fit(X, y)

# ------------------------
# Save model and preprocessor
# ------------------------
with open("preprocessor.pkl", "wb") as f:
    pickle.dump(preproc, f)

with open("logreg_tfidf_model.pkl", "wb") as f:
    pickle.dump(final_clf, f)

print("\nFinal model and preprocessor saved to 'logreg_tfidf_model.pkl' and 'preprocessor.pkl'")
