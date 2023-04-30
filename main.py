import numpy as np
import ipywidgets as widgets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import shap
from lime import lime_tabular
import itertools
import os

def load_dataset(file_path):
    assert file_path.endswith('.csv'), "File must have .csv extension."
    
    try:
        dataset = pd.read_csv(file_path)
    except FileNotFoundError:
        print("File not found.")
        raise
    return dataset

def data_validation(dataset):
    assert dataset.shape[1] >= 2, "Dataset should have atleast two columns (one for features and one for labels)"
    assert dataset.shape[0] >= 1, "Dataset should have atleast one row."
    
    unique_labels = dataset.iloc[:, -1].nunique()
    assert unique_labels >= 2, f"Dataset should contain atleast two unique labels. Detected: {unique_labels}"
    
    if dataset.isnull().any().any():
        print("Missing values detected. Please provide a dataset with no missing values.")
    else:
        return dataset

def preprocess_data(dataset):
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    if not np.array_equal(y, y.astype(int)):
        print("Non-integer labels detected.")
        raise
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler

def preprocess_new_data(scaler, new_data):
    X_new = scaler.transform(new_data)
    return X_new

def feature_elimination(X_train, y_train, n_features_to_select):
    total_features = X_train.shape[1]
    assert 1 <= n_features_to_select <= total_features, f"n_features_to_select should be in the range [1, {total_features}]"
    
    rfe = RFE(estimator=RandomForestClassifier(n_estimators=100, random_state=42), n_features_to_select=n_features_to_select)
    rfe.fit(X_train, y_train)
    return rfe

def train_incremental_model(X_train, y_train, n_estimators=100, n_splits=5):
    assert n_splits > 1, "n_splits should be greater than 1."
    
    train_splits = np.array_split(X_train, n_splits)
    target_splits = np.array_split(y_train, n_splits)
    
    classifiers = []
    for i in range(n_splits):
        part_classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        part_classifier.fit(train_splits[i], target_splits[i])
        classifiers.append(part_classifier)
    return classifiers

def predict_incremental_model(classifiers, X_test):
    predictions = []
    for clf in classifiers:
        part_preds = clf.predict_proba(X_test)
        predictions.append(part_preds)
    
    ensemble_preds = np.mean(predictions, axis=0)
    ensemble_preds = np.argmax(ensemble_preds, axis=1)
    return ensemble_preds

def model_explainability(classifiers, X_train):
    explainer = shap.TreeExplainer(classifiers[0])
    shap_values = explainer(X_train)
    shap.summary_plot(shap_values, X_train, plot_type='bar')

def evaluate_model(y_test, y_pred):
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("Accuracy score: ", accuracy_score(y_test, y_pred))
    print("Precision score:", precision_score(y_test, y_pred, average='weighted'))
    print("Recall score:", recall_score(y_test, y_pred, average='weighted'))
    print("F1 score:", f1_score(y_test, y_pred, average='weighted'))
    print("ROC-AUC score:", roc_auc_score(y_test, y_pred, multi_class='ovr', average='weighted'))

def plot_confusion_matrix(cm, classes):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def display_results(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, np.unique(y_test))

def grid_search(X_train, y_train):
    param_grid = {
        'n_estimators': [10, 50, 100, 200],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_

def run_pipeline(file_path, n_features_to_select, n_splits=5):
    dataset = load_dataset(file_path)
    dataset = data_validation(dataset)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(dataset)

    print("Grid search for hyperparameter tuning:")
    best_params = grid_search(X_train, y_train)
    print("Best parameters found: ", best_params)

    print("Feature Elimination (RFE):")
    rfe = feature_elimination(X_train, y_train, n_features_to_select)
    print("Selected features: ", rfe.support_)
    print("Feature rankings: ", rfe.ranking_)

    classifiers = train_incremental_model(X_train, y_train, n_estimators=best_params['n_estimators'], n_splits=n_splits)
    y_pred = predict_incremental_model(classifiers, X_test)

    print("Model Evaluation:")
    evaluate_model(y_test, y_pred)
    print("Visualization: Confusion Matrix")
    display_results(y_test, y_pred)

    print("Model Explainability:")
    model_explainability(classifiers, X_train)

    return classifiers, scaler

def predict_new_patient(new_patient_path, classifiers, scaler):
    new_data = load_dataset(new_patient_path)
    X_new = preprocess_new_data(scaler, new_data)
    new_pred = predict_incremental_model(classifiers, X_new)
    return new_pred

file_path_text = widgets.Text(
    value='',
    placeholder='Enter your medical dataset file path',
    description='File path:',
    disabled=False
)

n_features_widget = widgets.IntSlider(
    value=5,
    min=1,
    max=20,
    step=1,
    description='Top N Features:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
)

n_splits_widget = widgets.IntSlider(
    value=5,
    min=2,
    max=10,
    step=1,
    description='Incremental Splits:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
)

new_patient_path_text = widgets.Text(
    value='',
    placeholder="Enter new patient's CSV file path",
    description='New Patient:',
    disabled=False
)

run_button = widgets.Button(
    description='Run analysis',
    disabled=False,
    button_style='',
    tooltip='Click to run analysis'
)

predict_button = widgets.Button(
    description='Predict new patient',
    disabled=True,
    button_style='',
    tooltip='Click to predict new patient'
)

def on_run_button_click(b):
    file_path = file_path_text.value
    n_features_to_select = n_features_widget.value
    n_splits = n_splits_widget.value

    classifiers, scaler = run_pipeline(file_path, n_features_to_select, n_splits)
    b.model_data = {'classifiers': classifiers, 'scaler': scaler}
    predict_button.disabled = False

def on_predict_button_click(b):
    new_patient_path = new_patient_path_text.value
    classifiers = run_button.model_data['classifiers']
    scaler = run_button.model_data['scaler']
    predictions = predict_new_patient(new_patient_path, classifiers, scaler)
    print("Prediction for new patient(s):", predictions)

run_button.on_click(on_run_button_click)
predict_button.on_click(on_predict_button_click)

widgets.VBox([file_path_text, n_features_widget, n_splits_widget, run_button, new_patient_path_text, predict_button])
