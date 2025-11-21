# log_reg_tuning.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocessing import PreprocessorNoText, load_and_preprocess_no_text

X, y, groups, preproc = load_and_preprocess_no_text("train.csv")

C_values = [0.01, 0.1, 0.5, 1, 5]
penalties = ['l2','l1']
solvers = ['lbfgs','saga']

results = []

for C in C_values:
    for penalty in penalties:
        for solver in solvers:
            if penalty=='l1' and solver!='saga':
                continue
            gkf = GroupKFold(n_splits=5)
            fold_acc, fold_prec, fold_rec, fold_f1 = [], [], [], []
            for train_idx, val_idx in gkf.split(X,y,groups):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                clf = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=5000)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_val)

                fold_acc.append(accuracy_score(y_val,y_pred))
                fold_prec.append(precision_score(y_val,y_pred,average='macro'))
                fold_rec.append(recall_score(y_val,y_pred,average='macro'))
                fold_f1.append(f1_score(y_val,y_pred,average='macro'))

            mean_metrics = {
                'C': C,
                'penalty': penalty,
                'solver': solver,
                'mean_acc': np.mean(fold_acc),
                'mean_prec': np.mean(fold_prec),
                'mean_rec': np.mean(fold_rec),
                'mean_f1': np.mean(fold_f1)
            }
            results.append(mean_metrics)
            print(mean_metrics)

# Plot C vs metrics
metrics = ['mean_acc','mean_prec','mean_rec','mean_f1']
for metric in metrics:
    plt.plot(C_values, [r[metric] for r in results if r['penalty']=='l1' and r['solver']=='saga'], marker='o', label=metric)
plt.xlabel('C')
plt.ylabel('Score')
plt.title('Effect of C (L1/Saga) with polynomial features')
plt.legend()
plt.show()
