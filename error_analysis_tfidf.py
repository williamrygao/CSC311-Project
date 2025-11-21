# error_analysis.py
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# ------------------------
# Constants
# ------------------------
INDEX_TO_LABEL = ["ChatGPT", "Claude", "Gemini"]

# ------------------------
# Load model and preprocessor
# ------------------------
with open("preprocessor.pkl", "rb") as f:
    preproc = pickle.load(f)

with open("logreg_tfidf_model.pkl", "rb") as f:
    model = pickle.load(f)

# ------------------------
# Load test data
# ------------------------
test_df = pd.read_csv("test.csv")  # should contain same columns as train.csv
y_true = test_df["label"].map({"ChatGPT": 0, "Claude": 1, "Gemini": 2}).values

# Transform features
X_test = preproc.transform(test_df)

# ------------------------
# Predictions
# ------------------------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)
confidence = y_proba.max(axis=1)

# ------------------------
# Confusion matrix
# ------------------------
def save_confusion_matrix(y_true, y_pred, class_names=INDEX_TO_LABEL, save_path="writing/images/logistic_regression/confusion_matrix.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Logistic Regression (with TF-IDF) Confusion Matrix")
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Confusion matrix saved to {save_path}")

save_confusion_matrix(y_true, y_pred)

# ------------------------
# Classification report
# ------------------------
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=INDEX_TO_LABEL))

# ------------------------
# High-confidence misclassifications
# ------------------------
wrong_idx = np.where(y_pred != y_true)[0]
wrong_examples = test_df.iloc[wrong_idx].copy()
wrong_examples["true_label"] = [INDEX_TO_LABEL[i] for i in y_true[wrong_idx]]
wrong_examples["pred_label"] = [INDEX_TO_LABEL[i] for i in y_pred[wrong_idx]]
wrong_examples["confidence"] = confidence[wrong_idx]

top_wrong = wrong_examples.sort_values("confidence", ascending=False).head(10)
print("\nTop high-confidence misclassifications:")
print(top_wrong[["true_label", "pred_label", "confidence"]])

# ------------------------
# Feature importance for each class
# ------------------------
coef = model.coef_  # shape (n_classes, n_features)

# Map indices to feature names from preprocessor (TF-IDF)
if hasattr(preproc, "vectorizer"):
    feature_names = preproc.vectorizer.get_feature_names_out()
else:
    # fallback
    feature_names = [f"feat_{i}" for i in range(X_test.shape[1])]

for i, class_name in enumerate(INDEX_TO_LABEL):
    top_pos_idx = np.argsort(coef[i])[-10:][::-1]
    top_neg_idx = np.argsort(coef[i])[:10]
    
    print(f"\nClass {class_name} top positive features:")
    for idx in top_pos_idx:
        print(f"feat_{idx} ({feature_names[idx]}): {coef[i][idx]:.3f}")
    
    print(f"\nClass {class_name} top negative features:")
    for idx in top_neg_idx:
        print(f"feat_{idx} ({feature_names[idx]}): {coef[i][idx]:.3f}")
