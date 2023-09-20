import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier


# current cases = 780

df = pd.read_csv('dataset1.csv')

X = df.drop("Outcome", axis=1)  # Features
y = df["Outcome"]  # Target labels

cols = df.columns.tolist()

# Plot histograms for each feature based on the outcome (you can keep this for data exploration)
for label in cols[:-1]:
    plt.hist(df[df["Outcome"] == 1][label], color='blue', label='diabetes', alpha=0.7, density=True)
    plt.hist(df[df["Outcome"] == 0][label], color='green', label='no diabetes', alpha=0.7, density=True)
    plt.xlabel(label)
    plt.ylabel('Density')
    plt.legend()
    ##plt.show()

train, valid, test = np.split(df.sample(frac=1), (int(0.6 * len(df)), int(0.08 * len(df))))

def split_scale_data(dataframe, oversample=False):
    if dataframe.empty:
        return np.array([]), np.array([]), np.array([])

    x = dataframe[df.columns[:-1]].values
    y = dataframe[df.columns[-1]].values

    if not x.size:
        return np.array([]), np.array([]), np.array([])

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    if oversample:
        ros = RandomOverSampler(sampling_strategy='auto')
        x, y = ros.fit_resample(x, y)

    # y reshapen want y is een 1d array en moet een 2d array worden:
    y = y.reshape(-1, 1)
    data = np.hstack((x, y))  # Corrected the hstack call
    return data, x, y

train, x_train, y_train = split_scale_data(train, oversample=True)
valid, x_valid, y_valid = split_scale_data(valid, oversample=False)
test, x_test, y_test = split_scale_data(test, oversample=False)

if not x_train.size or not x_valid.size or not x_test.size:
    print("One of the data splits is empty.")
else:
    print("Data splits are not empty.")

kNN_model = KNeighborsClassifier(n_neighbors=1)

kNN_model.fit(x_train, y_train.ravel())
y_pred = kNN_model.predict(x_test)
print("k nearest neigh bor")
print(classification_report(y_test, y_pred))


regression_model = LogisticRegression()
regression_model.fit(x_train,y_train.ravel())

y_pred = regression_model.predict(x_test)
print("logistic regression")
print(classification_report(y_test,y_pred))



naive_bayes_model = GaussianNB()
naive_bayes_model.fit(x_train,y_train.ravel())
y_pred=naive_bayes_model.predict(x_test)

print("naive bayes")
print(classification_report(y_test,y_pred))

from sklearn.svm import SVC

svm_model = SVC()
svm_model.fit(x_train, y_train.ravel())
y_pred_svm = svm_model.predict(x_test)

print("SVM Classification Report:")
print(classification_report(y_test, y_pred_svm))


def ensemble_Allmodels(knn_model, regression_model, naive_bayes_model, x_train, y_train, x_test, y_test):
    # Create a soft voting ensemble
    ensemble = VotingClassifier(estimators=[
        ('knn', knn_model),
        ('logistic', regression_model),
        ('naive_bayes', naive_bayes_model)
    ], voting='soft')

    ensemble.fit(x_train, y_train.ravel())
    
    # Make predictions using the ensemble
    y_pred_ensemble = ensemble.predict(x_test)
    
    print("Ensemble (Soft Voting) Classification Report:\n", classification_report(y_test, y_pred_ensemble))
    return y_pred_ensemble

ensembled_model = ensemble_Allmodels(kNN_model,regression_model,naive_bayes_model,x_train,y_train,x_test,y_test)

