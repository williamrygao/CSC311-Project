# logistic_regression.py
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocessing import PreprocessorNoText, load_and_preprocess_no_text

# Load data
X, y, groups, preproc = load_and_preprocess_no_text("train.csv")

# Hyperparameters
C = 0.1
penalty = 'l1'
solver = 'saga'
max_iter = 5000

# Grouped CV
gkf = GroupKFold(n_splits=5)
fold_acc, fold_prec, fold_rec, fold_f1 = [], [], [], []

print("Grouped 5-fold CV with polynomial feature mapping:")
for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    clf = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=max_iter)
    clf.fit(X_train, y_train)
    y_val_pred = clf.predict(X_val)

    fold_acc.append(accuracy_score(y_val, y_val_pred))
    fold_prec.append(precision_score(y_val, y_val_pred, average='macro'))
    fold_rec.append(recall_score(y_val, y_val_pred, average='macro'))
    fold_f1.append(f1_score(y_val, y_val_pred, average='macro'))

    print(f"Fold {fold}: Acc={fold_acc[-1]:.3f}, Prec={fold_prec[-1]:.3f}, Recall={fold_rec[-1]:.3f}, F1={fold_f1[-1]:.3f}")

print("\nMean CV metrics:")
print(f"Accuracy={np.mean(fold_acc):.3f}, Precision={np.mean(fold_prec):.3f}, Recall={np.mean(fold_rec):.3f}, F1={np.mean(fold_f1):.3f}")

# Train final model
final_clf = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=max_iter)
final_clf.fit(X, y)

# Save model and preprocessor
with open("preprocessor_no_text.pkl", "wb") as f:
    pickle.dump(preproc, f)
with open("logreg_no_text_model.pkl", "wb") as f:
    pickle.dump(final_clf, f)

print("\nSaved preprocessor and model.")
