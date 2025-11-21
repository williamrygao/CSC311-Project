# logistic_regression_no_text.py
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from preprocessing import load_and_preprocess  # your existing preprocessing (numerical + categorical)

# ------------------------
# Load processed training data (ignoring text features)
# ------------------------
X, y, groups, preproc = load_and_preprocess("train.csv", return_preprocessor=True)
# X.shape = (n_samples, n_features), y.shape = (n_samples,), groups.shape = (n_samples,)

# Save the fitted preprocessor for later use on test data
with open("preprocessor_no_text.pkl", "wb") as f:
    pickle.dump(preproc, f)

# ------------------------
# Grouped 5-fold cross-validation
# ------------------------
gkf = GroupKFold(n_splits=5)
fold_acc = []
fold_metrics = []

print("Grouped 5-fold cross-validation (ignoring text):")

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    clf = LogisticRegression(solver="lbfgs", max_iter=1000)
    clf.fit(X_train, y_train)

    # Predictions
    y_train_pred = clf.predict(X_train)
    y_val_pred = clf.predict(X_val)

    # Metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_val_pred, average="macro")

    fold_acc.append(val_acc)
    fold_metrics.append((precision, recall, f1))

    print(f"Fold {fold}: Train acc = {train_acc:.3f}, Val acc = {val_acc:.3f}, "
          f"Precision = {precision:.3f}, Recall = {recall:.3f}, F1 = {f1:.3f}")

# ------------------------
# Mean metrics across folds
# ------------------------
mean_val_acc = np.mean(fold_acc)
mean_precision = np.mean([m[0] for m in fold_metrics])
mean_recall = np.mean([m[1] for m in fold_metrics])
mean_f1 = np.mean([m[2] for m in fold_metrics])

print("\nMean validation metrics across folds:")
print(f"Accuracy = {mean_val_acc:.3f}, Precision = {mean_precision:.3f}, "
      f"Recall = {mean_recall:.3f}, F1 = {mean_f1:.3f}")

# ------------------------
# Train final model on all training data
# ------------------------
final_clf = LogisticRegression(solver="lbfgs", max_iter=1000)
final_clf.fit(X, y)

# Save parameters for pred.py
np.savez("lr_params_no_text.npz", W=final_clf.coef_, b=final_clf.intercept_)
print("\nFinal model trained and parameters saved to lr_params_no_text.npz")
