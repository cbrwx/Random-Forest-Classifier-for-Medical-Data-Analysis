# Random-Forest-Classifier-for-Medical-Data-Analysis
This code is a Random Forest Classifier for analyzing medical data. It can be used to preprocess, validate, and classify medical data using the Random Forest algorithm. The code uses Python, Numpy, Pandas, Scikit-learn, Matplotlib, and SHAP packages.

It is designed to handle medical datasets, perform feature elimination, and optimize the model through hyperparameter tuning. By dividing the training process into smaller increments, it mitigates the risk of overlooking issues in the data. With an intuitive IPython widget-based interface, users can easily input their dataset, select the number of top features, and specify the number of incremental splits for training. The code also evaluates the performance of the model using metrics such as accuracy, precision, recall, F1 score, and ROC-AUC score, giving users an understanding of the model's reliability. Additionally, the model explainability is provided using SHAP values, allowing users to gain insights into the importance of individual features. Finally, the code enables users to input new patient data and predict their illness using the trained model. This comprehensive approach ensures accurate and reliable predictions while minimizing the chances of overlooking potential issues in the data.

# How to Use
- Provide the path of the CSV file containing your medical data.
- Select the number of top features to use in feature elimination (RFE) with the slider.
- Select the number of incremental splits to use in training with the slider.
- Click the 'Run analysis' button to train the model and display results.
- Provide the path of the CSV file containing new patient data to predict.
- Click the 'Predict new patient' button to predict the new patient data.
# CSV File Format
The CSV file should contain columns of features and a label column with integer labels (0 or 1). There should be no missing values in the data.

# Output
The output of the code includes:

- Grid search for hyperparameter tuning
- Feature Elimination (RFE)
- Model Evaluation metrics (accuracy, precision, recall, f1-score, ROC-AUC)
- Visualization of the Confusion Matrix
- Model Explainability using SHAP values
- Prediction for new patient(s)
# Code Explanation
The code consists of several functions for data validation, preprocessing, feature elimination, incremental training, model explainability, and evaluation. It also includes a pipeline function to connect these functions and output the results. The code uses IPython widgets to provide a user-friendly interface for the pipeline.

- The load_dataset() function reads the CSV file and returns a Pandas DataFrame. The data_validation() function checks for any missing values in the dataset and returns the validated dataset.

- The preprocess_data() function performs standard scaling and train-test splitting of the dataset and returns the preprocessed data and the scaler object. The preprocess_new_data() function takes the scaler object and new patient data as input and returns the preprocessed new patient data.

- The feature_elimination() function uses Recursive Feature Elimination (RFE) with the Random Forest algorithm to select the top N features to use in training.

- The train_incremental_model() function trains multiple classifiers using incremental training and returns the classifiers. The predict_incremental_model() function takes the classifiers and new data as input and returns the predictions.

- The model_explainability() function uses SHAP values to explain the model predictions.

- The evaluate_model() function calculates various evaluation metrics (accuracy, precision, recall, f1-score, ROC-AUC) for the model and prints the results.

- The plot_confusion_matrix() and display_results() functions display the Confusion Matrix visualization.

- The grid_search() function uses Grid Search to find the best hyperparameters for the Random Forest Classifier.

- The run_pipeline() function connects the above functions and outputs the results.

- The predict_new_patient() function takes the new patient data, classifiers, and scaler as input and returns the prediction for the new patient.

- The IPython widgets provide a user-friendly interface for the pipeline, allowing the user to provide the CSV file paths and select the number of top features and incremental splits.

This Random Forest Classifier code tries to provide an efficient way to preprocess, validate, and classify medical data. The incremental training and feature elimination techniques improve the performance of the model. The SHAP values explain the model predictions, making it more interpretable. The interface is easily customizable and modifiable for different datasets.
