"""
Naive Bayes Classifier with Maximum Likelihood Estimation
For CSC311 Machine Learning Project

Now includes:
- Grouped 5-fold cross-validation (grouped by student_id)
- Hyperparameter tuning for Laplace smoothing (alpha)
- Metric plots: Accuracy, Precision, Recall, F1
"""

import numpy as np
import pandas as pd
import re
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer


class NaiveBayesClassifier:
    """Multinomial Naive Bayes with Laplace smoothing for categorical and numerical features"""

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes = None
        self.class_priors = {}
        self.feature_likelihoods = {}
        self.numerical_params = {}

    def fit(self, X, y, feature_types):
        self.classes = np.unique(y)
        n_samples = len(y)

        # Compute class priors
        for cls in self.classes:
            self.class_priors[cls] = np.sum(y == cls) / n_samples

        # Compute likelihoods
        for cls in self.classes:
            mask = (y == cls)
            X_cls = X[mask]
            self.feature_likelihoods[cls] = {}
            self.numerical_params[cls] = {}

            for f_idx in range(X.shape[1]):
                f_type = feature_types.get(f_idx, 'categorical')

                if f_type == 'categorical':
                    values = np.unique(X[:, f_idx])
                    counts = {}
                    for val in values:
                        counts[val] = (np.sum(X_cls[:, f_idx] == val) + self.alpha) / \
                                      (len(X_cls) + self.alpha * len(values))
                    self.feature_likelihoods[cls][f_idx] = counts
                else:  # numerical
                    vals = X_cls[:, f_idx].astype(float)
                    mean = np.mean(vals)
                    std = np.std(vals) + 1e-6
                    self.numerical_params[cls][f_idx] = {'mean': mean, 'std': std}

    def _gaussian_prob(self, x, mean, std):
        return (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((x - mean) ** 2) / (2 * std ** 2))

    def predict(self, X, feature_types):
        preds = []
        for sample in X:
            log_probs = {}
            for cls in self.classes:
                log_prob = np.log(self.class_priors[cls])
                for f_idx in range(len(sample)):
                    f_type = feature_types.get(f_idx, 'categorical')
                    val = sample[f_idx]

                    if f_type == 'categorical':
                        likelihood = self.feature_likelihoods[cls][f_idx].get(val,
                                                                              self.alpha / (
                                                                                          self.alpha * (
                                                                                              len(
                                                                                                  self.feature_likelihoods[
                                                                                                      cls][
                                                                                                      f_idx]) + 1)))
                        log_prob += np.log(likelihood + 1e-10)
                    else:
                        mean = self.numerical_params[cls][f_idx]['mean']
                        std = self.numerical_params[cls][f_idx]['std']
                        log_prob += np.log(self._gaussian_prob(float(val), mean, std) + 1e-10)
                log_probs[cls] = log_prob
            preds.append(max(log_probs, key=log_probs.get))
        return np.array(preds)


def extract_rating(s):
    if pd.isna(s) or s == '':
        return None
    m = re.match(r'^(\d+)', str(s))
    return int(m.group(1)) if m else None


def process_multiselect(series, target_tasks):
    processed = []
    for s in series:
        if pd.isna(s) or s == '':
            processed.append([])
        else:
            processed.append([t for t in target_tasks if t in str(s)])
    return processed


def prepare_features(df):
    target_tasks = [
        'Math computations',
        'Writing or debugging code',
        'Data processing or analysis',
        'Explaining complex concepts simply',
    ]

    best_lists = process_multiselect(df['Which types of tasks do you feel this model handles best? (Select all that apply.)'], target_tasks)
    subopt_lists = process_multiselect(df['For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)'], target_tasks)

    mlb_best = MultiLabelBinarizer()
    mlb_subopt = MultiLabelBinarizer()
    best_encoded = mlb_best.fit_transform(best_lists)
    subopt_encoded = mlb_subopt.fit_transform(subopt_lists)

    # Numerical features
    numeric_feats = [
        df['How likely are you to use this model for academic tasks?'].apply(extract_rating),
        df['Based on your experience, how often has this model given you a response that felt suboptimal?'].apply(extract_rating),
        df['How often do you expect this model to provide responses with references or supporting evidence?'].apply(extract_rating),
        df["How often do you verify this model's responses?"].apply(extract_rating)
    ]
    feature_list = []
    feature_names = []
    feature_types = {}
    idx = 0

    for feat, name in zip(numeric_feats, ['academic', 'subopt', 'reference', 'verify']):
        feature_list.append(feat.values.reshape(-1, 1))
        feature_names.append(name)
        feature_types[idx] = 'numerical'
        idx += 1

    feature_list.append(best_encoded)
    for t in mlb_best.classes_:
        feature_names.append(f'best_{t}')
        feature_types[idx] = 'categorical'
        idx += 1

    feature_list.append(subopt_encoded)
    for t in mlb_subopt.classes_:
        feature_names.append(f'subopt_{t}')
        feature_types[idx] = 'categorical'
        idx += 1

    X = np.hstack(feature_list)
    y = df['label'].values

    return X, y, feature_types, feature_names


def evaluate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


def main():
    # Load dataset
    df = pd.read_csv('training_data_clean.csv')
    df = df.dropna()
    X, y, feature_types, feature_names = prepare_features(df)

    # Get student_ids for grouping
    if 'student_id' not in df.columns:
        raise ValueError("Dataset must contain 'student_id' column for grouped CV")
    groups = df['student_id'].values

    alphas = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    cv_results = []

    # 5-fold grouped cross-validation
    gkf = GroupKFold(n_splits=5)
    for alpha in alphas:
        fold_metrics = []
        for train_idx, val_idx in gkf.split(X, y, groups):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            nb = NaiveBayesClassifier(alpha=alpha)
            nb.fit(X_train, y_train, feature_types)
            y_val_pred = nb.predict(X_val, feature_types)
            metrics = evaluate_metrics(y_val, y_val_pred)
            fold_metrics.append(metrics)

        # Average metrics over folds
        avg_metrics = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0].keys()}
        avg_metrics['alpha'] = alpha
        cv_results.append(avg_metrics)
        print(f"Alpha {alpha}: {avg_metrics}")

    # Convert results to DataFrame
    results_df = pd.DataFrame(cv_results)
    results_df.to_csv('grouped_cv_results.csv', index=False)

    # Plot metrics vs alpha
    plt.figure(figsize=(8, 5))
    plt.plot(results_df['alpha'], results_df['accuracy'], marker='o', label='Accuracy')
    plt.plot(results_df['alpha'], results_df['precision'], marker='o', label='Precision')
    plt.plot(results_df['alpha'], results_df['recall'], marker='o', label='Recall')
    plt.plot(results_df['alpha'], results_df['f1'], marker='o', label='F1')
    plt.xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('Metric')
    plt.title('Naive Bayes Hyperparameter Tuning (Grouped CV)')
    plt.legend()
    plt.grid(True)
    plt.savefig('alpha_metrics_plot.png')
    plt.show()


if __name__ == "__main__":
    main()
