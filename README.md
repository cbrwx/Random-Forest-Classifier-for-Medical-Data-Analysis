# Random-Forest-Classifier-for-Medical-Data-Analysis
This code is a Random Forest Classifier for analyzing medical data. It can be used to preprocess, validate, and classify medical data using the Random Forest algorithm. The code uses Python, Numpy, Pandas, Scikit-learn, Matplotlib, and SHAP packages.

Designed specifically for medical data analysis. Leveraging the capabilities of Python, Numpy, Pandas, Scikit-learn, Matplotlib, and SHAP packages, this solution offers robust preprocessing, validation, and classification capabilities using the Random Forest algorithm.

Key Features:

- Medical dataset handling: The code is tailored to manage medical datasets, ensuring accurate analysis and predictions.
- Feature elimination: Utilizing Recursive Feature Elimination (RFE), the model identifies the most relevant features for improved performance.
- Hyperparameter tuning: Grid search optimization fine-tunes the model, resulting in more reliable predictions.
- Incremental training: By dividing the training process into smaller increments, the code minimizes the risk of overlooking data issues.
- Intuitive interface: The IPython widget-based interface allows users to effortlessly input their dataset, choose the number of top features, and set incremental splits for training.
- Performance evaluation: The model's reliability is assessed using metrics such as accuracy, precision, recall, F1 score, and ROC-AUC score, providing users with valuable insights into its performance.
- Model explainability: SHAP values highlight the importance of individual features, helping users understand the model's decision-making process.
- New patient data prediction: Users can input new patient data and predict their illness using the trained model, ensuring practical real-world applications.
- This comprehensive solution ensures accurate and reliable predictions, empowering users to tackle medical data analysis challenges while minimizing the risk of overlooking potential issues in the data.

# How to Use
- Provide the path of the CSV file containing your medical data.
- Select the number of top features to use in feature elimination (RFE) with the slider.
- Select the number of incremental splits to use in training with the slider.
- Click the 'Run analysis' button to train the model and display results.
- Provide the path of the CSV file containing new patient data to predict.
- Click the 'Predict new patient' button to predict the new patient data.

# CSV File Format
The CSV file should contain columns of features and a label column with integer labels (0 or 1). There should be no missing values in the data.

Here's an example of a simplified CSV file for a real-world medical dataset. This dataset contains information about patients and whether they have diabetes or not. The dataset has the following columns:

- Age: The age of the patient
- BMI: The patient's Body Mass Index
- Glucose: The patient's fasting blood glucose level
- BloodPressure: The patient's systolic blood pressure
- Insulin: The patient's insulin level
- Diabetes: The target variable - whether the patient has diabetes (1) or not (0)
Example CSV file content:
```
Age,BMI,Glucose,BloodPressure,Insulin,Diabetes
42,23.5,99,75,140,0
36,26.1,112,85,195,1
29,28.2,130,95,205,1
57,35.1,150,100,240,1
61,32.4,140,85,265,1
45,24.3,108,70,155,0
38,29.5,120,80,190,1
34,27.7,110,90,175,0
49,25.2,120,72,180,1
52,33.1,140,92,230,1
```
Please note that this is a small example dataset and does not represent the full complexity of real-world medical data. When working with actual data, you may encounter more complex features, a larger number of features, and more instances. Additionally, it is essential to preprocess the data (handle missing values, normalize, etc.) before using it in machine learning models.

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

.cbrwx
