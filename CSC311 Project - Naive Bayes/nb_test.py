# nb_test.py
import pandas as pd
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from nb import NaiveBayesClassifier, prepare_features

# -------------------------
# Load test data
# -------------------------
df_test = pd.read_csv('test.csv')
df_test = df_test.dropna()

# -------------------------
# Load trained model
# -------------------------
with open('naive_bayes_model.json', 'r') as f:
    model_data = json.load(f)

alpha = model_data['alpha']
nb_clf = NaiveBayesClassifier(alpha=alpha)

# Load model parameters
nb_clf.classes = np.array(model_data['classes'])
nb_clf.class_priors = {k: float(v) for k, v in model_data['class_priors'].items()}

# Feature likelihoods
nb_clf.feature_likelihoods = {}
for cls, feat_dict in model_data['feature_likelihoods'].items():
    nb_clf.feature_likelihoods[cls] = {}
    for feat_idx, val_dict in feat_dict.items():
        nb_clf.feature_likelihoods[cls][int(feat_idx)] = {k: float(v) for k, v in val_dict.items()}

# Numerical parameters
nb_clf.numerical_params = {}
for cls, feat_dict in model_data['numerical_params'].items():
    nb_clf.numerical_params[cls] = {}
    for feat_idx, params in feat_dict.items():
        nb_clf.numerical_params[cls][int(feat_idx)] = {'mean': float(params['mean']), 'std': float(params['std'])}

# -------------------------
# Prepare features
# -------------------------
X_test, y_test, feature_types, feature_names = prepare_features(df_test)

# -------------------------
# Predict
# -------------------------
y_pred = nb_clf.predict(X_test, feature_types)

# -------------------------
# Evaluate
# -------------------------
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, labels=nb_clf.classes)

macro_precision = np.mean(precision)
macro_recall = np.mean(recall)
macro_f1 = np.mean(f1)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall: {macro_recall:.4f}")
print(f"Macro F1: {macro_f1:.4f}")

print("\nPer-class metrics:")
for i, cls in enumerate(nb_clf.classes):
    print(f"\n{cls}:")
    print(f"  Precision: {precision[i]:.4f}")
    print(f"  Recall: {recall[i]:.4f}")
    print(f"  F1: {f1[i]:.4f}")
    print(f"  Support: {support[i]}")

# -------------------------
# Confusion matrix
# -------------------------
cm = confusion_matrix(y_test, y_pred, labels=nb_clf.classes)
print("\nConfusion Matrix (rows=true, cols=predicted):")
print(f"{'':<15} {' '.join(f'{cls:<12}' for cls in nb_clf.classes)}")
for i, cls in enumerate(nb_clf.classes):
    print(f"{cls:<15} {' '.join(f'{cm[i,j]:<12}' for j in range(len(nb_clf.classes)))}")
