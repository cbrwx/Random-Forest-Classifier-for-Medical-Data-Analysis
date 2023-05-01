# Predicts the presence of heart disease using the UCI Heart Disease dataset. 
# The script employs Logistic Regression as the machine learning model to classify individuals as having 
# heart disease or not, cbrwx.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_dataset(url):
    column_names = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
    ]
    data = pd.read_csv(url, header=None, names=column_names, na_values="?")

    return data

def data_validation(dataset):
    is_valid = dataset.shape[0] > 0 and dataset.shape[1] > 1 and not dataset.isnull().values.any()
    return is_valid

def preprocess_data(dataset):
    data = dataset.dropna()
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    y = np.where(y > 0, 1, 0)  # Convert diagnosis to binary (0 = no disease, 1 = disease)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler

def train_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def test_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

def preprocess_new_data(scaler, new_data):
    return scaler.transform(new_data)

def predict(model, X_new):
    return model.predict(X_new)

def plot_confusion_matrix(cm, classes):
    sns.heatmap(cm, annot=True, fmt='d', cmap="YlGnBu", xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def main():
    # Load the data
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    data = load_dataset(url)
    if not data_validation(data):
        raise ValueError("Dataset failed validation")

    # Prepare the data for training
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data)

    # Train the model
    model = train_model(X_train, y_train)

    # Test the model
    test_model(model, X_test, y_test)

    # Load the separate CSV file
    separate_csv = "path/to/separate/csv"
    new_data = load_dataset(separate_csv)

    # Prepare the new data for prediction
    X_new = preprocess_new_data(scaler, new_data.iloc[:, :-1].values)

    # Make predictions on the new data
    y_new_pred = predict(model, X_new)

    # Print the predictions
    print("Predictions for the separate CSV file:", y_new_pred)

    # Plot confusion matrix
    cm = confusion_matrix(y_test, model.predict(X_test)) 
    class_labels = np.unique(y_test) 
    plot_confusion_matrix(cm, class_labels)

if __name__ == "__main__":
    main()
