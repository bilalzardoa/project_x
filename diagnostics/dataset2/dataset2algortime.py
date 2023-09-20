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

if dataset is not None:
    # Now you can work with the 'dataset' which contains the data from the CSV file
    print(dataset)  # Example: Display the first few rows of the data

df = dataset

#train, valid, test = np.split(df.sample(frac=1), (int(0.6 * len(df)), int(0.08 * len(df))))

train_percent = 0.67
valid_percent = 0.18  # You can adjust this based on your needs
test_percent = 0.15   # You can adjust this based on your needs

# First, split into train and temp sets
train, temp = train_test_split(df, test_size=1 - train_percent, random_state=42)

# Then, split the temp set into validation and test sets
valid, test = train_test_split(temp, test_size=test_percent / (test_percent + valid_percent), random_state=42)

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

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

# Check the lengths of the training set after oversampling
#print(len(x_train))
#print(len(y_train))

# Create a StandardScaler instance and fit it on the training data


knn_model = KNeighborsClassifier(n_neighbors=2,weights='distance')
knn_model.fit(x_train, y_train.ravel())
y_pred = knn_model.predict(x_test)
#print("k nearest neighbor")
#print(classification_report(y_test, y_pred))


regression_model = LogisticRegression(class_weight='balanced')
regression_model.fit(x_train,y_train.ravel())

y_pred = regression_model.predict(x_test)
#print("logistic regression")
#print(classification_report(y_test,y_pred))


training_model = VotingClassifier(estimators=[
    ('knn', knn_model),
    ('logistic', regression_model)
], voting='soft')

# Train the ensemble on your training data
training_model.fit(x_train, y_train.ravel())

# Make predictions using the ensemble
y_pred_ensemble = training_model.predict(x_test)

# Evaluate the ensemble's performance
print("Ensemble (Soft Voting) Classification Report:\n", classification_report(y_test, y_pred_ensemble))






def rank_features_with_random_forest(dataset, n_estimators=100, top_features=10, random_state=42):
    # Split the dataset into features and target variable
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values.ravel()

    # Create and train a Random Forest model
    random_forest_model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    random_forest_model.fit(X, y)

    # Get feature importances
    feature_importances = random_forest_model.feature_importances_

    # Get feature names
    feature_names = dataset.columns[:-1]

    # Create a ranked list of features
    feature_importance_ranking = sorted(zip(feature_names, feature_importances), key=lambda x: x[1], reverse=True)

    # Visualize the top features
    top_feature_names, top_feature_importances = zip(*feature_importance_ranking[:top_features])

    plt.figure(figsize=(10, 6))
    plt.barh(range(top_features), top_feature_importances, align='center')
    plt.yticks(range(top_features), top_feature_names)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Top Feature Importances')
    #plt.show()

    return feature_importance_ranking


feature_ranking = rank_features_with_random_forest(dataset)
#print(feature_ranking)

def discretize_a1c(df):
    # Define bin edges and labels
    bin_edges = [0,4,5.6,6.4,14]
    bin_labels = ['low', 'normal', 'high','diabetes']

    # Create a new column with the discretized values
    df['A1c Category'] = pd.cut(df['hemoglobine_a1c'], bins=bin_edges, labels=bin_labels)

    return df

# Sample DataFrame with 'Hemoglobine A1c' column
data = dataset['hemoglobine_a1c']
print(data)
df = pd.DataFrame(data)

# Apply the discretization method
df = discretize_a1c(df)
print(df)
print()
# Apply one-hot encoding to the 'A1c Category' column
df = pd.get_dummies(df, columns=['A1c Category'], prefix=['A1c'])
# Print the resulting DataFrame
print(df)

# Use the updated DataFrame as features in your ensemble model
x_ensemble = df.values  # Convert to a NumPy array
y_ensemble = dataset['diagnose'].values  # Assuming 'diagnosis' is your target variable

# Train the kNN model on your training data
kNN_model = KNeighborsClassifier(n_neighbors=2, weights='distance')
kNN_model.fit(x_ensemble, y_ensemble.ravel())

# Train the logistic regression model on your training data
#regression_model = LogisticRegression(class_weight='balanced')
#regression_model.fit(x_ensemble, y_ensemble.ravel())


svm_model = SVC(class_weight='balanced', probability=True)  # Use probability=True for soft voting
svm_model.fit(x_ensemble, y_ensemble.ravel())

# Create an ensemble model
hemoglobine_a1c_model = VotingClassifier(estimators=[
    ('knn', kNN_model),
    ('logistic', svm_model)
], voting='soft')

# Train the ensemble on your training data
hemoglobine_a1c_model.fit(x_ensemble, y_ensemble.ravel())

# Make predictions using the ensemble
y_pred_ensemble = hemoglobine_a1c_model.predict(x_ensemble)

# Evaluate the ensemble's performance
print()
print("Hemoglobine A1c Classification Report:\n", classification_report(y_ensemble, y_pred_ensemble))

models = [
    ('Hemoglobine A1c Model', hemoglobine_a1c_model),  # Replace with your Hemoglobine A1c-based model
    ('Training Model', training_model)  # Your regular model
]

# Create a voting ensemble with soft voting
traing_en_hemoglobine_a1c_model = VotingClassifier(estimators=models, voting='soft')

# Train the ensemble on your training data
traing_en_hemoglobine_a1c_model.fit(x_train, y_train.ravel())

js_model = traing_en_hemoglobine_a1c_model.fit(x_train, y_train.ravel())

y_pred_ensemble = traing_en_hemoglobine_a1c_model.predict(x_test)
print("traing_en_hemoglobine_a1c_model Classification Report:\n", classification_report(y_test, y_pred_ensemble))


import os
import joblib

directory_path = "diagnostics/dataset2/"

# Create the directory if it doesn't exist
if not os.path.exists(directory_path):
    os.makedirs(directory_path)

# Save the ensemble model in the specified directory with the desired filename
joblib.dump(traing_en_hemoglobine_a1c_model, os.path.join(directory_path, "saved_model.pkl"))


# Load the ensemble model from the file
loaded_model = joblib.load("diagnostics/dataset2/saved_model.pkl")
#y_pred_ensemble = loaded_model.predict(x_test)
print("loaded_model Classification Report:\n", classification_report(y_test, y_pred_ensemble))


from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback  # Import traceback for detailed error information
import pandas as pd  # Import pandas for handling missing values

import numpy as np

#modelP.predict()
#print("modelP Classification Report:\n", classification_report(y_test, y_pred_ensemble))


# Define a function to convert nested NumPy arrays to lists
def nested_array_to_list(arr):
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    elif isinstance(arr, list):
        return [nested_array_to_list(item) for item in arr]
    else:
        return arr

app = Flask(__name__)
CORS(app)  # Allow CORS for all routes

import json
@app.route('/p', methods=['POST'])
def predict():
    try:
        # Get the input data from the request
        input_data = request.get_json()

        if not isinstance(input_data, list):
            return jsonify({'error': 'Input data should be a list of lists'}), 400

        # Convert the input data to a NumPy array
        input_array = np.array(input_data, dtype=float)

        # Use the fitted scaler from your training phase
        processed_data = scaler.transform(input_array)  # Apply the same scaler used during training

        input_array = np.array(input_array).reshape(1, -1)  # Reshape to make it a single sample

        ia = nested_array_to_list(input_array)
        pdl = processed_data.tolist()

        xt = nested_array_to_list(x_test)

        modelP = traing_en_hemoglobine_a1c_model.predict(processed_data)

        if modelP == 1:
            diagnose = "positive"
        else:
            diagnose = "negative"

        return jsonify({"diagnose": diagnose, "inputdata": input_data, "inputArray": ia, "processed data": pdl, "x test": xt})
   
    except Exception as e:
        # Log the error details for debugging
        traceback.print_exc()  # Print detailed error information to console
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)  # Enable Flask's debug mode for detailed error messages



from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(traing_en_hemoglobine_a1c_model, x_train, y_train.ravel(), cv=5, scoring='accuracy')

# Print cross-validation scores
print("Cross-Validation Scores:", cv_scores)

# Calculate the mean and standard deviation of the scores
mean_score = cv_scores.mean()
std_score = cv_scores.std()

print("Mean Accuracy:", mean_score)
print("Standard Deviation:", std_score)














import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

# Train the ensemble on your training data
traing_en_hemoglobine_a1c_model.fit(x_train, y_train.ravel())

# Predict using the trained model
y_pred_ensemble = traing_en_hemoglobine_a1c_model.predict(x_test)

# Calculate classification report and accuracy
classification_rep = classification_report(y_test, y_pred_ensemble, output_dict=True)
accuracy = accuracy_score(y_test, y_pred_ensemble)

# Extract precision, recall, and calculate f1-score for each class
labels = [label for label in classification_rep.keys() if label.isdigit()]
precision = [classification_rep[label]['precision'] for label in labels]
recall = [classification_rep[label]['recall'] for label in labels]
f1_score = [2 * (p * r) / (p + r) if (p + r) > 0 else 0 for p, r in zip(precision, recall)]

# Create a custom color palette
custom_colors = ['#000000', '#8B4513', '#000000', '#8B4513']

# Create a bar chart
fig, ax = plt.subplots(figsize=(10, 6))

bar_width = 0.2
index = np.arange(len(labels))

bars = []  # List to store all bars

for i, category in enumerate(['Precision', 'Recall', 'F1-Score']):
    color = custom_colors[i]  # Use custom color for each category
    bar = plt.bar(index + bar_width * i, [precision, recall, f1_score][i], bar_width, label=category, align='center', alpha=0.7, color=color)
    bars.append(bar)

# Add accuracy as a separate bar
bar4 = plt.bar(index + bar_width * 3, [accuracy] * len(labels), bar_width, label='Accuracy', align='center', alpha=0.7, color=custom_colors[3])
bars.append(bar4)

plt.xlabel('Classes')
plt.ylabel('Scores')
plt.title('Classification Report Scores by Class')
plt.xticks(index + bar_width * 1.5, labels)
plt.legend()

plt.tight_layout()
#plt.show()






