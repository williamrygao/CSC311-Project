import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

file_name = "training_data_clean.csv"

def extract_rating(response):
    match = re.match(r'^(\d+)', str(response))
    return int(match.group(1)) if match else None

def main():
    df = pd.read_csv(file_name)
    df.dropna(inplace=True)

    # Numerical features (Likert-scale)
    numerical_features = [
        'How likely are you to use this model for academic tasks?',
        'Based on your experience, how often has this model given you a response that felt suboptimal?',
        'How often do you expect this model to provide responses with references or supporting evidence?',
        "How often do you verify this model's responses?"
    ]

    # Labels for Likert-scale features
    academic_labels = {
        1: '1 - Not at all likely',
        2: '2 - Slightly likely',
        3: '3 - Neutral / Unsure',
        4: '4 - Likely',
        5: '5 - Very likely'
    }

    frequency_labels = {
        1: '1 - Never',
        2: '2 - Rarely',
        3: '3 - Sometimes',
        4: '4 - Often',
        5: '5 - Very Often'
    }

    # Extract numeric ratings
    numeric_data = pd.DataFrame()
    for feature in numerical_features:
        numeric_data[feature] = df[feature].apply(extract_rating)

    # Plot distributions
    for feature in numerical_features:
        plt.figure(figsize=(6,4))
        plt.hist(numeric_data[feature].dropna(), bins=np.arange(1,7)-0.5, edgecolor='black')

        # Choose correct labels
        if feature == 'How likely are you to use this model for academic tasks?':
            labels = [academic_labels[i] for i in range(1,6)]
        else:
            labels = [frequency_labels[i] for i in range(1,6)]

        plt.xticks(range(1,6), labels, rotation=30, ha='right')  # number + word
        plt.xlabel('Rating')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {feature}', wrap=True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
