# Milestone 1: Data Preprocessing for Online Payment Fraud Detection
![image](https://github.com/user-attachments/assets/609a8224-c285-4a5b-819f-5fff7af20c22)

## Overview

This repository contains the code and documentation for Milestone 1 of the project focusing on the preprocessing of the online payment fraudulent detection dataset. The objective of this milestone is to prepare the dataset for subsequent analysis and modeling by performing necessary preprocessing steps.

## Preprocessing Steps

1. **Load Dataset**: Load the raw dataset from a CSV file into a pandas DataFrame.
2. **Initial Inspection**: Conduct a preliminary examination of the dataset to understand its structure and summarize the statistics of numerical features.
3. **Handle Missing Values**: Identify and handle missing values by filling numerical features with their median and categorical features with their mode.
4. **Remove Duplicates**: Check for and eliminate any duplicate entries in the dataset.
5. **Visualize Transaction Types**: Create visual representations to understand the distribution of different transaction types.
6. **Check Dataset Balance**: Analyze the balance between fraudulent and non-fraudulent transactions to assess the need for resampling techniques.
7. **Convert Categorical Features**: Transform categorical features into numerical representations suitable for machine learning models.
8. **Normalize Numerical Features**: Scale numerical features to a standard range to improve model performance.
9. **Save Final Preprocessed Dataset**: Save the processed dataset for use in subsequent phases of the project.
10. **Store & Retrieve Data in AWS S3**: Develop a mechanism for storing and retrieving the preprocessed dataset from AWS S3.

# Milestone 2: Model Training and Evaluation for Online Payment Fraud Detection

![image]([https://github.com/user-attachments/assets/609a8224-c285-4a5b-819f-5fff7af20c22](https://www.cardinalpeak.com/blog/best-practices-when-training-machine-learning-models))

## Overview

This repository contains the code and documentation for Milestone 2 of the project focusing on training machine learning models and evaluating their performance for online payment fraud detection. The objective of this milestone is to build predictive models that can effectively classify transactions as fraudulent or non-fraudulent.

## Model Training Steps

1. **Load Preprocessed Dataset**: Load the cleaned and preprocessed dataset prepared in Milestone 1.
2. **Split Dataset**: Divide the dataset into training and testing sets to ensure the model is evaluated on unseen data.
3. **Choose Models**: Select appropriate machine learning algorithms for fraud detection (e.g., Logistic Regression, Random Forest, XGBoost).
4. **Hyperparameter Tuning**: Optimize model parameters using techniques such as Grid Search or Random Search to enhance model performance.
5. **Train Models**: Train the selected models on the training dataset and evaluate their performance using appropriate metrics.
6. **Evaluate Models**: Assess model performance on the test dataset using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.
7. **Model Comparison**: Compare the performance of different models to determine the best-performing model for fraud detection.
8. **Save Trained Models**: Persist the trained models using joblib or pickle for future use in deployment or further analysis.
