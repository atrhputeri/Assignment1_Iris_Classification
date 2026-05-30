# Iris Classification using Spark MLlib

## Overview

In this project, I performed a classification task on the Iris dataset using Apache Spark MLlib. The objective was to classify iris flower species based on sepal and petal measurements using machine learning algorithms.

The Iris dataset contains 150 flower records with the following attributes:

a) Sepal Length
b) Sepal Width
c) Petal Length
d) Petal Width
e) Species

The dataset was loaded into a Spark DataFrame and processed using Spark MLlib for machine learning analysis.

## Description of Dataset and Methodology

The workflow of this project includes:

1. Loading the Iris dataset using SparkSession.
2. Data preprocessing:
   * Duplicate checking and removal
   * Missing value checking
   * Label encoding using StringIndexer
   * Feature vector creation using VectorAssembler
3. Splitting the dataset into training and testing sets (80:20 ratio).
4. Training three classification models:
   * Decision Tree
   * Random Forest
   * Logistic Regression
5. Hyperparameter tuning for all models using:
   * Cross-validation
   * Grid search
6. Model evaluation using:
   * Accuracy
   * Precision
   * Recall
   * F1-score
7. Comparing model performance and identifying the best-performing classifier.

## Summary of Results and Key Findings

Three classification models were developed and evaluated using accuracy, precision, recall, and F1-score. Hyperparameter tuning was done using cross-validation and grid search to improve model performance.

The results show that Random Forest performed the best with 1.0 for all evaluation metrics. Decision Tree and Logistic Regression both achieved similar results with 95.83% accuracy, and their precision, recall, and F1-score are also the same.

Overall, Random Forest gave the best performance compared to the other two models based on the evaluation results. This shows that ensemble methods can give better results for this dataset.

In conclusion, Apache Spark MLlib is useful for building and testing machine learning models like classification in a simple and efficient way.

## Instructions to Reproduce the Analysis

1. Install Apache Spark and PySpark.
2. Download the Iris dataset and save it as `iris.csv`.
3. Open the Jupyter Notebook containing the project code.
4. Ensure the dataset file is located in the same directory as the notebook.
5. Run all notebook cells from top to bottom.
6. Review the evaluation metrics generated for Decision Tree, Random Forest, and Logistic Regression models.
7. Compare the results to identify the best-performing model.
