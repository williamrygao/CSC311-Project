# log_reg_tfidf_test.py
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ------------------------
# Load fitted Preprocessor
# ------------------------
with open("preprocessor.pkl", "rb") as f:
    preproc = pickle.load(f)

# ------------------------
# Load trained logistic regression parameters
# ------------------------
params = np.load("lr_params.npz")
W, b = params["W"], params["b"]  # W shape (n_classes, n_features)

# ------------------------
# Load test data
# ------------------------
test_df = pd.read_csv("test.csv")
y_test = test_df["label"].map({"ChatGPT":0, "Claude":1, "Gemini":2}).values

# ------------------------
# Preprocess test features
# ------------------------
X_test = preproc.transform(test_df)  # uses fitted scaler and TF-IDF

# ------------------------
# Predict
# ------------------------
logits = X_test @ W.T + b          # shape (n_samples, n_classes)
y_pred = np.argmax(logits, axis=1)

# ------------------------
# Evaluate
# ------------------------
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro')
rec = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print(f"Test set metrics:")
print(f"Accuracy = {acc:.3f}")
print(f"Precision = {prec:.3f}")
print(f"Recall = {rec:.3f}")
print(f"F1 score = {f1:.3f}")
