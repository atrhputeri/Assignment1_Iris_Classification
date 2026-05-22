# Iris Classification using Spark MLlib
Assignment 1

Overview
________________________________________

In this project, I performed a classification task on Iris dataset using Apache Spark MLlib. 
The goal is to classify iris flower species based on sepal and petal measurements using machine learning algorithms.

The Iris dataset contains 150 flower records with the following attributes:
a) Sepal Length
b) Sepal Width
c) Petal Length
d) Petal Width
e) Species

The dataset was loaded into a Spark DataFrame for preprocessing and machine learning tasks.

Description of dataset and methodology
_______________________________________

The workflow of this project includes:

1. Data loading using SparkSession
2. Data preprocessing (Duplicate checking)
3. Splitting dataset into training and testing sets
4. Training classification models:
  a) Decision Tree
  b) Random Forest
  c) Logistic Regression
5. Hyperparameter tuning using:
  a) Cross-validation
  b) Grid search
6. Model evaluation using:
  a) Accuracy
  b) Precision
  c) Recall
  d) F1-score

Summary of results and key findings
________________________________________________

In this project, three classification models were used. Decision Tree, Random Forest, and Logistic Regression. The models were tested using accuracy, precision, recall, and F1-score. The results show that Decision Tree and Random Forest gave the best performance with an accuracy of 95.83%. Logistic Regression gave lower performance with an accuracy of 79.17%. Then I run model tuning using cross-validation and grid search and applied to the Random Forest model. The tuned Random Forest model also gave strong and stable results, showing that it is the most suitable model for this dataset.

Overall, tree-based models performed better than Logistic Regression for the Iris dataset, and Spark MLlib worked well for building and testing machine learning classification models. 

