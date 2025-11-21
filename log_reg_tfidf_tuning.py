# tune_tfidf_logreg.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocessing_tfidf import Preprocessor

# ------------------------
# Load data
# ------------------------
df = pd.read_csv("train.csv")
y = df["label"].map({"ChatGPT":0, "Claude":1, "Gemini":2}).values
groups = df["student_id"].values

# ------------------------
# Hyperparameter grid
# ------------------------
tfidf_features = [100, 300, 500, 700]  # added 700
C_values = [0.1, 0.5, 1, 5, 10]         # added 0.5 and 5
penalties = ['l2', 'l1']
solvers = ['lbfgs', 'saga']
ngram_ranges = [(1,1), (1,2)]           # test unigrams and bigrams

param_grid = []
for max_feat in tfidf_features:
    for ngram in ngram_ranges:
        for C in C_values:
            for penalty in penalties:
                for solver in solvers:
                    if penalty == 'l1' and solver != 'saga':
                        continue
                    param_grid.append({
                        'max_features': max_feat,
                        'C': C,
                        'penalty': penalty,
                        'solver': solver,
                        'ngram_range': ngram
                    })

# ------------------------
# Run grouped 5-fold CV
# ------------------------
results = []
gkf = GroupKFold(n_splits=5)

for params in param_grid:
    preproc = Preprocessor(max_features=params['max_features'], ngram_range=params['ngram_range'])
    X = preproc.fit_transform(df)

    fold_acc, fold_prec, fold_rec, fold_f1 = [], [], [], []

    for train_idx, val_idx in gkf.split(X, y, groups):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        clf = LogisticRegression(
            C=params['C'],
            penalty=params['penalty'],
            solver=params['solver'],
            max_iter=5000
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)

        fold_acc.append(accuracy_score(y_val, y_pred))
        fold_prec.append(precision_score(y_val, y_pred, average='macro'))
        fold_rec.append(recall_score(y_val, y_pred, average='macro'))
        fold_f1.append(f1_score(y_val, y_pred, average='macro'))

    results.append({
        'params': params,
        'mean_acc': np.mean(fold_acc),
        'mean_prec': np.mean(fold_prec),
        'mean_rec': np.mean(fold_rec),
        'mean_f1': np.mean(fold_f1)
    })

    print(f"TF-IDF={params['max_features']}, ngram={params['ngram_range']}, C={params['C']}, "
          f"penalty={params['penalty']}, solver={params['solver']} -> "
          f"Acc={np.mean(fold_acc):.3f}, Prec={np.mean(fold_prec):.3f}, "
          f"Recall={np.mean(fold_rec):.3f}, F1={np.mean(fold_f1):.3f}")

# ------------------------
# Best hyperparameters
# ------------------------
best_result = max(results, key=lambda x: x['mean_f1'])
best_params = best_result['params']
print("\nBest hyperparameters:", best_params)
print(f"Mean metrics: Acc={best_result['mean_acc']:.3f}, Prec={best_result['mean_prec']:.3f}, "
      f"Recall={best_result['mean_rec']:.3f}, F1={best_result['mean_f1']:.3f}")

# ------------------------
# Plotting function
# ------------------------
def plot_metrics(results, hyperparam, fixed_params=None):
    import matplotlib.pyplot as plt

    metrics = ['mean_acc', 'mean_prec', 'mean_rec', 'mean_f1']
    names = ['Accuracy', 'Precision', 'Recall', 'F1']

    if fixed_params is None:
        fixed_params = {}

    # get all unique values for this hyperparameter
    values = sorted(list(set(r['params'][hyperparam] for r in results)), key=lambda x: str(x))
    xvals = [str(v) for v in values]  # convert everything to string for plotting

    plt.figure(figsize=(8,5))
    for metric, name in zip(metrics, names):
        yvals = []
        for val in values:
            # filter results by hyperparameter and fixed params
            filtered = [r for r in results if r['params'][hyperparam] == val]
            for k, v in fixed_params.items():
                filtered = [r for r in filtered if r['params'][k] == v]
            if filtered:
                # pick the best F1 for this value
                best = max(filtered, key=lambda x: x['mean_f1'])
                yvals.append(best[metric])
            else:
                yvals.append(np.nan)
        plt.plot(xvals, yvals, marker='o', label=name)

    plt.xlabel(hyperparam)
    plt.ylabel("Score")
    plt.title(f"Effect of {hyperparam} on metrics")
    plt.grid(True)
    plt.legend()
    plt.show()

# ------------------------
# Plot each hyperparameter independently
# ------------------------
other_params = {k:v for k,v in best_params.items()}

plot_metrics(results, 'max_features', {k:v for k,v in other_params.items() if k!='max_features'})
plot_metrics(results, 'C', {k:v for k,v in other_params.items() if k!='C'})
plot_metrics(results, 'penalty', {k:v for k,v in other_params.items() if k!='penalty'})
plot_metrics(results, 'solver', {k:v for k,v in other_params.items() if k!='solver'})
plot_metrics(results, 'ngram_range', {k:v for k,v in other_params.items() if k!='ngram_range'})
