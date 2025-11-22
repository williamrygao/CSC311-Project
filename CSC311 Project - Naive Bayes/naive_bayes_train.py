"""
Naive Bayes Classifier with Maximum Likelihood Estimation
For CSC311 Machine Learning Project
"""

import numpy as np
import pandas as pd
import re
import json
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer


class NaiveBayesClassifier:
    """
    Multinomial Naive Bayes with Laplace smoothing for categorical and numerical features
    """

    def __init__(self, alpha=1.0):
        """
        Initialize Naive Bayes classifier

        Parameters:
        -----------
        alpha : float
            Laplace smoothing parameter (default=1.0)
        """
        self.alpha = alpha
        self.classes = None
        self.class_priors = {}
        self.feature_likelihoods = {}
        self.numerical_params = {}  # For Gaussian features

    def fit(self, X, y, feature_types):
        """
        Train the Naive Bayes classifier

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target labels
        feature_types : dict
            Dictionary mapping feature indices to types ('categorical' or 'numerical')
        """
        self.classes = np.unique(y)
        n_samples = len(y)

        # Calculate class priors P(C)
        for cls in self.classes:
            self.class_priors[cls] = np.sum(y == cls) / n_samples

        # Calculate feature likelihoods P(X|C)
        for cls in self.classes:
            cls_mask = (y == cls)
            X_cls = X[cls_mask]

            self.feature_likelihoods[cls] = {}
            self.numerical_params[cls] = {}

            for feature_idx in range(X.shape[1]):
                feature_type = feature_types.get(feature_idx, 'categorical')

                if feature_type == 'categorical':
                    # For categorical features, use frequency counts with Laplace smoothing
                    feature_values = X_cls[:, feature_idx]
                    unique_values = np.unique(X[:, feature_idx])  # All possible values

                    value_counts = {}
                    for val in unique_values:
                        count = np.sum(feature_values == val)
                        # Laplace smoothing
                        value_counts[val] = (count + self.alpha) / (
                                    len(feature_values) + self.alpha * len(unique_values))

                    self.feature_likelihoods[cls][feature_idx] = value_counts

                elif feature_type == 'numerical':
                    # For numerical features, assume Gaussian distribution
                    feature_values = X_cls[:, feature_idx].astype(float)
                    mean = np.mean(feature_values)
                    std = np.std(feature_values) + 1e-6  # Add small value to avoid division by zero

                    self.numerical_params[cls][feature_idx] = {'mean': mean, 'std': std}

    def _gaussian_probability(self, x, mean, std):
        """Calculate Gaussian probability density"""
        exponent = np.exp(-((x - mean) ** 2) / (2 * std ** 2))
        return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

    def predict_proba(self, X, feature_types):
        """
        Predict class probabilities for X

        Returns:
        --------
        probabilities : dict
            Dictionary mapping class labels to log probabilities
        """
        predictions = []

        for sample in X:
            log_probs = {}

            for cls in self.classes:
                # Start with log prior
                log_prob = np.log(self.class_priors[cls])

                # Add log likelihoods for each feature
                for feature_idx in range(len(sample)):
                    feature_type = feature_types.get(feature_idx, 'categorical')

                    if feature_type == 'categorical':
                        feature_val = sample[feature_idx]
                        # Get likelihood, use smoothing for unseen values
                        if feature_val in self.feature_likelihoods[cls][feature_idx]:
                            likelihood = self.feature_likelihoods[cls][feature_idx][feature_val]
                        else:
                            # Unseen value - use smoothed probability
                            n_values = len(self.feature_likelihoods[cls][feature_idx])
                            likelihood = self.alpha / (self.alpha * (n_values + 1))

                        log_prob += np.log(likelihood + 1e-10)

                    elif feature_type == 'numerical':
                        feature_val = float(sample[feature_idx])
                        mean = self.numerical_params[cls][feature_idx]['mean']
                        std = self.numerical_params[cls][feature_idx]['std']

                        prob = self._gaussian_probability(feature_val, mean, std)
                        log_prob += np.log(prob + 1e-10)

                log_probs[cls] = log_prob

            predictions.append(log_probs)

        return predictions

    def predict(self, X, feature_types):
        """
        Predict class labels for X

        Returns:
        --------
        predictions : array-like
            Predicted class labels
        """
        proba = self.predict_proba(X, feature_types)
        predictions = [max(prob_dict, key=prob_dict.get) for prob_dict in proba]
        return np.array(predictions)

    def save_model(self, filename):
        """Save model parameters to JSON file"""
        model_data = {
            'alpha': self.alpha,
            'classes': self.classes.tolist(),
            'class_priors': self.class_priors,
            'feature_likelihoods': {
                str(cls): {
                    str(feat_idx): {str(k): float(v) for k, v in feat_dict.items()}
                    for feat_idx, feat_dict in cls_dict.items()
                }
                for cls, cls_dict in self.feature_likelihoods.items()
            },
            'numerical_params': {
                str(cls): {
                    str(feat_idx): {'mean': float(params['mean']), 'std': float(params['std'])}
                    for feat_idx, params in cls_dict.items()
                }
                for cls, cls_dict in self.numerical_params.items()
            }
        }

        with open(filename, 'w') as f:
            json.dump(model_data, f, indent=2)


def extract_rating(response):
    """Extract numeric rating from responses like '3 - Sometimes'"""
    if pd.isna(response) or response == '':
        return None
    match = re.match(r'^(\d+)', str(response))
    return int(match.group(1)) if match else None


def process_multiselect(series, target_tasks):
    """Convert multiselect strings to lists, keeping only specified features"""
    processed = []
    for response in series:
        if pd.isna(response) or response == '':
            processed.append([])
        else:
            present_tasks = [task for task in target_tasks if task in str(response)]
            processed.append(present_tasks)
    return processed


def prepare_features(df):
    """
    Prepare features from the dataset

    Returns:
    --------
    X : numpy array
        Feature matrix
    y : numpy array
        Labels
    feature_types : dict
        Dictionary mapping feature indices to types
    feature_names : list
        Names of features
    """
    # Define the tasks we want to use as features
    target_tasks = [
        'Math computations',
        'Writing or debugging code',
        'Data processing or analysis',
        'Explaining complex concepts simply',
    ]

    # Process multi-select columns
    best_tasks_lists = process_multiselect(
        df['Which types of tasks do you feel this model handles best? (Select all that apply.)'],
        target_tasks
    )
    suboptimal_tasks_lists = process_multiselect(
        df[
            'For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)'],
        target_tasks
    )

    # Encode multi-select features
    mlb_best = MultiLabelBinarizer()
    mlb_subopt = MultiLabelBinarizer()

    best_tasks_encoded = mlb_best.fit_transform(best_tasks_lists)
    suboptimal_tasks_encoded = mlb_subopt.fit_transform(suboptimal_tasks_lists)

    # Extract numerical features
    academic_numeric = df['How likely are you to use this model for academic tasks?'].apply(extract_rating)
    subopt_numeric = df[
        'Based on your experience, how often has this model given you a response that felt suboptimal?'].apply(
        extract_rating)
    reference_numeric = df[
        'How often do you expect this model to provide responses with references or supporting evidence?'].apply(
        extract_rating)
    verify_numeric = df["How often do you verify this model's responses?"].apply(extract_rating)

    # Combine all features
    feature_list = []
    feature_names = []
    feature_types = {}
    current_idx = 0

    # Add numerical features
    for feat, name in [(academic_numeric, 'academic_likelihood'),
                       (subopt_numeric, 'subopt_frequency'),
                       (reference_numeric, 'reference_expectation'),
                       (verify_numeric, 'verify_frequency')]:
        feature_list.append(feat.values.reshape(-1, 1))
        feature_names.append(name)
        feature_types[current_idx] = 'numerical'
        current_idx += 1

    # Add binary features from multi-select
    feature_list.append(best_tasks_encoded)
    for task in mlb_best.classes_:
        feature_names.append(f'best_{task}')
        feature_types[current_idx] = 'categorical'
        current_idx += 1

    feature_list.append(suboptimal_tasks_encoded)
    for task in mlb_subopt.classes_:
        feature_names.append(f'subopt_{task}')
        feature_types[current_idx] = 'categorical'
        current_idx += 1

    X = np.hstack(feature_list)
    y = df['label'].values

    return X, y, feature_types, feature_names


def evaluate_model(y_true, y_pred, classes):
    """
    Evaluate model performance

    Returns:
    --------
    metrics : dict
        Dictionary containing accuracy, precision, recall, and F1 scores
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, labels=classes)

    # Compute macro averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)

    metrics = {
        'accuracy': accuracy,
        'precision_per_class': dict(zip(classes, precision)),
        'recall_per_class': dict(zip(classes, recall)),
        'f1_per_class': dict(zip(classes, f1)),
        'support_per_class': dict(zip(classes, support)),
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1
    }

    return metrics


def main():
    """Main training pipeline"""

    # Load data
    print("Loading data...")
    df = pd.read_csv('training_data_clean.csv')

    # Drop rows with missing values
    print(f"Original dataset size: {len(df)}")
    df = df.dropna()
    print(f"Dataset size after dropping NaN: {len(df)}")

    # Prepare features
    print("\nPreparing features...")
    X, y, feature_types, feature_names = prepare_features(df)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of samples: {len(y)}")
    print(f"Feature names: {feature_names}")

    # Split data: first 70% for train+val, last 30% for test
    n_samples = len(X)
    n_train_val = int(0.85 * n_samples)

    X_train_val = X[:n_train_val]
    y_train_val = y[:n_train_val]
    X_test = X[n_train_val:]
    y_test = y[n_train_val:]

    # Further split train_val: first 50% overall for train, next 20% for validation
    n_train = int(0.7 * n_samples)
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_val = X[n_train:n_train_val]
    y_val = y[n_train:n_train_val]

    print(f"\nData split:")
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Train Naive Bayes with different alpha values
    print("\n" + "=" * 50)
    print("Hyperparameter Tuning: Testing different alpha values")
    print("=" * 50)

    alphas = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    best_alpha = None
    best_val_acc = 0
    results = []

    for alpha in alphas:
        print(f"\nTesting alpha = {alpha}")
        nb = NaiveBayesClassifier(alpha=alpha)
        nb.fit(X_train, y_train, feature_types)

        # Evaluate on training set
        y_train_pred = nb.predict(X_train, feature_types)
        train_metrics = evaluate_model(y_train, y_train_pred, nb.classes)

        # Evaluate on validation set
        y_val_pred = nb.predict(X_val, feature_types)
        val_metrics = evaluate_model(y_val, y_val_pred, nb.classes)

        print(f"  Training accuracy: {train_metrics['accuracy']:.4f}")
        print(f"  Validation accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  Validation macro F1: {val_metrics['macro_f1']:.4f}")

        results.append({
            'alpha': alpha,
            'train_acc': train_metrics['accuracy'],
            'val_acc': val_metrics['accuracy'],
            'val_f1': val_metrics['macro_f1']
        })

        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_alpha = alpha

    print("\n" + "=" * 50)
    print(f"Best alpha: {best_alpha} with validation accuracy: {best_val_acc:.4f}")
    print("=" * 50)

    # Train final model with best alpha on full training+validation set
    print("\n\nTraining final model with best hyperparameters...")
    final_nb = NaiveBayesClassifier(alpha=best_alpha)
    final_nb.fit(X_train_val, y_train_val, feature_types)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    y_test_pred = final_nb.predict(X_test, feature_types)
    test_metrics = evaluate_model(y_test, y_test_pred, final_nb.classes)

    print("\n" + "=" * 50)
    print("FINAL TEST SET RESULTS")
    print("=" * 50)
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Macro Precision: {test_metrics['macro_precision']:.4f}")
    print(f"Macro Recall: {test_metrics['macro_recall']:.4f}")
    print(f"Macro F1: {test_metrics['macro_f1']:.4f}")

    print("\nPer-class metrics:")
    for cls in final_nb.classes:
        print(f"\n{cls}:")
        print(f"  Precision: {test_metrics['precision_per_class'][cls]:.4f}")
        print(f"  Recall: {test_metrics['recall_per_class'][cls]:.4f}")
        print(f"  F1: {test_metrics['f1_per_class'][cls]:.4f}")
        print(f"  Support: {test_metrics['support_per_class'][cls]}")

    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_test_pred, labels=final_nb.classes)
    print("Predicted ->")
    print(f"{'True':<15} {' '.join(f'{cls:<12}' for cls in final_nb.classes)}")
    for i, cls in enumerate(final_nb.classes):
        print(f"{cls:<15} {' '.join(f'{cm[i, j]:<12}' for j in range(len(final_nb.classes)))}")

    # Save the model
    print("\nSaving model...")
    final_nb.save_model('naive_bayes_model.json')

    # Also save feature information
    feature_info = {
        'feature_names': feature_names,
        'feature_types': {str(k): v for k, v in feature_types.items()},
        'best_alpha': best_alpha
    }
    with open('feature_info.json', 'w') as f:
        json.dump(feature_info, f, indent=2)

    print("\nModel and feature information saved to /mnt/user-data/outputs/")

    # Save results summary
    results_df = pd.DataFrame(results)
    results_df.to_csv('hyperparameter_results.csv', index=False)
    print("Hyperparameter tuning results saved to /mnt/user-data/outputs/hyperparameter_results.csv")

    return final_nb, feature_types, feature_names, test_metrics


if __name__ == "__main__":
    main()
