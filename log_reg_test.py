# log_reg_test.py
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocessing import PreprocessorNoText

# Load preprocessor and model
with open("preprocessor_no_text.pkl","rb") as f:
    preproc = pickle.load(f)
with open("logreg_no_text_model.pkl","rb") as f:
    clf = pickle.load(f)

# Load test data
test_df = pd.read_csv("test.csv")
y_test = test_df["label"].map({"ChatGPT":0,"Claude":1,"Gemini":2}).values

# Transform features
X_test = preproc.transform(test_df)

# Predict
y_pred = clf.predict(X_test)

# Evaluate
acc = accuracy_score(y_test,y_pred)
prec = precision_score(y_test,y_pred,average='macro')
rec = recall_score(y_test,y_pred,average='macro')
f1 = f1_score(y_test,y_pred,average='macro')

print(f"Test metrics with polynomial features:")
print(f"Accuracy = {acc:.3f}")
print(f"Precision = {prec:.3f}")
print(f"Recall = {rec:.3f}")
print(f"F1 score = {f1:.3f}")
