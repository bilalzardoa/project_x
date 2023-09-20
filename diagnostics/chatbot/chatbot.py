import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

#dataperprocessing

def read_csv_to_variable(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Usage
csv_file_path = 'dataset2.csv' 
dataset = read_csv_to_variable(csv_file_path)


column_aliases = {
    'geslacht': ['Gender', 'geslacht', 'sex'],
    'diagnose': ['CLASS','outcome','resultaat'], 
    'cholesterol': ['cholesterol', 'chol'],
}

column_name_mapping = {
    'Gender': 'geslacht',
    'AGE': 'leeftijd',
    'Urea': 'ureum',
    'Cr': 'creatinine',
    'HbA1c': 'hemoglobine_a1c',
    'Chol': 'cholesterol',
    'TG': 'triglyceriden',
    'HDL': 'hdl_cholesterol',
    'LDL': 'ldl_cholesterol',
    'VLDL': 'vldl_cholesterol',
    'BMI': 'bmi',
    'CLASS': 'diagnose',
}

expected_features = [
    'geslacht', 'leeftijd', 'ureum', 'creatinine', 'hemoglobine_a1c',
    'cholesterol', 'triglyceriden', 'hdl_cholesterol', 'ldl_cholesterol',
    'vldl_cholesterol', 'bmi', 'diagnose'
]

# Rename columns using the mapping
dataset = dataset.rename(columns=column_name_mapping)
columns_to_ignore = ['ID', 'No_Pation']
all_columns = dataset.columns
# Define the columns to ignore by subtracting expected features from all columns
columns_to_ignore += [col for col in all_columns if col not in expected_features]

# Drop the columns to ignore from the dataset
dataset = dataset.drop(columns=columns_to_ignore)
# Handle column aliases
for actual_col_name, alias_list in column_aliases.items():
    for alias in alias_list:
        if alias in dataset.columns:
            dataset.rename(columns={alias: actual_col_name}, inplace=True)
            
# Handle missing values (you can customize this based on your strategy)
dataset = dataset.dropna()

label_encoder = LabelEncoder()
dataset['geslacht'] = label_encoder.fit_transform(dataset['geslacht'])

if len(dataset['diagnose'].unique()) == 2:
    # If 'CLASS' is already binary, no need to change anything
    pass
else:
    # Convert 'N', 'no', 'nee', 'negatief', etc. to 0, and everything else to 1 in the 'CLASS' column
    binary_mapping = {'N': 0, 'no': 0, 'nee': 0, 'negatief': 0}
    dataset['diagnose'] = dataset['diagnose'].map(lambda x: binary_mapping.get(x, 1))

df = dataset

train_percent = 0.67
valid_percent = 0.18  # You can adjust this based on your needs
test_percent = 0.15   # You can adjust this based on your needs

# First, split into train and temp sets
train, temp = train_test_split(df, test_size=1 - train_percent, random_state=42)

# Then, split the temp set into validation and test sets
valid, test = train_test_split(temp, test_size=test_percent / (test_percent + valid_percent), random_state=42)



##############################################################################################################
scaler = StandardScaler()

def split_scale_data(dataframe, oversample=False):

    if dataframe.empty:
        return [], [], []

    x = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values

    if not x.size:
        return [], [], []

    x = scaler.fit_transform(x)

    if oversample:
        ros = RandomOverSampler(sampling_strategy='auto')
        x, y = ros.fit_resample(x, y)

    # Reshape y to be a 2D array
    y = y.reshape(-1, 1)

    # Combine x and y horizontally
    data = list(x) + list(y)
    return data, x, y

# Oversample the minority class in the training set
train, x_train, y_train = split_scale_data(train, oversample=True)
valid, x_valid, y_valid = split_scale_data(valid, oversample=False)
test, x_test, y_test = split_scale_data(test, oversample=False)

import numpy as np

new_data = [1,22,1,3,44,15,1,5,6,4,1]
new_data = np.array(new_data).reshape(1, -1)  # Reshape to make it a single sample

new_data_scaled = scaler.transform(new_data)
print(new_data_scaled)