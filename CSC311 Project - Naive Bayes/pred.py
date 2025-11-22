"""
Prediction script for CSC311 Machine Learning Project
This script loads a trained Naive Bayes model and makes predictions
"""

import sys
import csv
import json
import re
import os
import numpy as np
import pandas as pd


def extract_rating(response):
    """Extract numeric rating from responses like '3 - Sometimes'"""
    if pd.isna(response) or response == '' or response is None:
        return None
    match = re.match(r'^(\d+)', str(response))
    return int(match.group(1)) if match else None


def process_multiselect_ordered(response, ordered_classes):
    """
    Convert multiselect string to binary vector using the EXACT order from training.

    This matches the order that MultiLabelBinarizer used during training,
    which is ALPHABETICAL, not the order specified in target_tasks.

    Parameters:
    -----------
    response : str
        The multi-select response string
    ordered_classes : list
        The exact order of classes from MultiLabelBinarizer.classes_ (alphabetical)

    Returns:
    --------
    binary_vector : list
        Binary vector in the same order as ordered_classes
    """
    if pd.isna(response) or response == '':
        return [0] * len(ordered_classes)

    binary_vector = []
    for task in ordered_classes:
        if task in str(response):
            binary_vector.append(1)
        else:
            binary_vector.append(0)

    return binary_vector


def gaussian_probability(x, mean, std):
    """Calculate Gaussian probability density"""
    exponent = np.exp(-((x - mean) ** 2) / (2 * std ** 2))
    return (1 / (np.sqrt(2 * np.pi) * std)) * exponent


def predict(row, model_data, feature_info):
    """
    Make prediction for a single row

    Parameters:
    -----------
    row : dict
        Dictionary containing feature values
    model_data : dict
        Trained model parameters
    feature_info : dict
        Feature configuration information

    Returns:
    --------
    prediction : str
        Predicted class label
    """
    # Extract model parameters
    classes = model_data['classes']
    class_priors = model_data['class_priors']
    feature_likelihoods = model_data['feature_likelihoods']
    numerical_params = model_data['numerical_params']
    alpha = model_data['alpha']
    feature_types = feature_info['feature_types']

    # CRITICAL: Use the CORRECT feature orderings from training
    # MultiLabelBinarizer creates features in ALPHABETICAL order, not target_tasks order!
    # This is the order that was used during training and saved in the model
    mlb_best_classes = [
        'Data processing or analysis',
        'Explaining complex concepts simply',
        'Math computations',
        'Writing or debugging code'
    ]

    mlb_subopt_classes = [
        'Data processing or analysis',
        'Explaining complex concepts simply',
        'Math computations',
        'Writing or debugging code'
    ]

    # Prepare features in the same order as training
    features = []

    # Numerical features (indices 0-3)
    academic = extract_rating(row.get('How likely are you to use this model for academic tasks?', ''))
    subopt = extract_rating(
        row.get('Based on your experience, how often has this model given you a response that felt suboptimal?', ''))
    reference = extract_rating(
        row.get('How often do you expect this model to provide responses with references or supporting evidence?', ''))
    verify = extract_rating(row.get("How often do you verify this model's responses?", ''))

    features.extend([academic, subopt, reference, verify])

    # Multi-select features - best tasks (indices 4-7)
    # MUST use the alphabetical order from MultiLabelBinarizer, not target_tasks order
    best_tasks_response = row.get('Which types of tasks do you feel this model handles best? (Select all that apply.)',
                                  '')
    best_tasks_binary = process_multiselect_ordered(best_tasks_response, mlb_best_classes)
    features.extend(best_tasks_binary)

    # Multi-select features - suboptimal tasks (indices 8-11)
    # MUST use the alphabetical order from MultiLabelBinarizer, not target_tasks order
    subopt_tasks_response = row.get(
        'For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)',
        '')
    subopt_tasks_binary = process_multiselect_ordered(subopt_tasks_response, mlb_subopt_classes)
    features.extend(subopt_tasks_binary)

    # Calculate log probabilities for each class
    log_probs = {}

    for cls in classes:
        # Start with log prior
        log_prob = np.log(class_priors[cls])

        # Add log likelihoods for each feature
        for feature_idx, feature_val in enumerate(features):
            feature_type = feature_types[str(feature_idx)]

            if feature_val is None:
                # Handle missing values - skip this feature
                continue

            if feature_type == 'categorical':
                # Categorical feature
                feature_val = int(feature_val)  # Convert to int for binary features

                if str(feature_idx) in feature_likelihoods[str(cls)]:
                    feat_dict = feature_likelihoods[str(cls)][str(feature_idx)]

                    if str(feature_val) in feat_dict:
                        likelihood = feat_dict[str(feature_val)]
                    else:
                        # Unseen value - use smoothed probability
                        n_values = len(feat_dict)
                        likelihood = alpha / (alpha * (n_values + 1))

                    log_prob += np.log(likelihood + 1e-10)

            elif feature_type == 'numerical':
                # Numerical feature
                feature_val = float(feature_val)

                if str(feature_idx) in numerical_params[str(cls)]:
                    params = numerical_params[str(cls)][str(feature_idx)]
                    mean = params['mean']
                    std = params['std']

                    prob = gaussian_probability(feature_val, mean, std)
                    log_prob += np.log(prob + 1e-10)

        log_probs[cls] = log_prob

    # Return class with highest log probability
    prediction = max(log_probs, key=log_probs.get)
    return prediction


def predict_all(filename):
    """
    Make predictions for all data in filename

    Parameters:
    -----------
    filename : str
        Name of CSV file containing test data

    Returns:
    --------
    predictions : list
        List of predicted class labels
    """
    # Load model and feature info
    # Look for model files in the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'naive_bayes_model.json')
    feature_path = os.path.join(script_dir, 'feature_info.json')

    try:
        with open(model_path, 'r') as f:
            model_data = json.load(f)

        with open(feature_path, 'r') as f:
            feature_info = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Model files not found.")
        print(f"Looking for files in: {script_dir}")
        print(f"Please ensure 'naive_bayes_model.json' and 'feature_info.json' are in the same directory as pred.py")
        raise e

    # Read the CSV file
    df = pd.read_csv(filename)

    # Make predictions
    predictions = []
    for idx, row in df.iterrows():
        pred = predict(row, model_data, feature_info)
        predictions.append(pred)

    return predictions
