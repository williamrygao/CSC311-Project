import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load the full dataset
df = pd.read_csv("training_data_clean.csv")

# ------------------------
# Split by student
# ------------------------
# Assumes there is a 'student_id' column
unique_students = df['student_id'].unique()

# Reserve 15% of students for test set
train_students, test_students = train_test_split(
    unique_students,
    test_size=0.15,
    random_state=42  # ensures reproducibility
)

# Masks for rows belonging to train/test students
train_mask = df['student_id'].isin(train_students)
test_mask = df['student_id'].isin(test_students)

df_train = df[train_mask].reset_index(drop=True)
df_test = df[test_mask].reset_index(drop=True)

print(f"Training samples: {len(df_train)}, Test samples: {len(df_test)}")
print(f"Training students: {len(train_students)}, Test students: {len(test_students)}")

# Save to separate CSVs
df_train.to_csv("train.csv", index=False)
df_test.to_csv("test.csv", index=False)
