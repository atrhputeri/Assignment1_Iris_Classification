# Iris Species Classification Using PySpark MLlib

## Project Overview

This project applies a machine learning classification workflow using **PySpark MLlib** to classify Iris flower species based on their physical measurements.

The project includes data cleaning, preprocessing, feature preparation, model training, hyperparameter tuning, evaluation, and comparison of three classification algorithms:

- Decision Tree
- Random Forest
- Logistic Regression

The models are trained to predict the species of Iris flowers:
- Setosa
- Versicolor
- Virginica

---

# Dataset

The dataset used in this project is the **Iris dataset**, provided in the repository. 

The dataset contains 150 original records with the following features:

| Feature | Description |
|---|---|
| sepal_length | Length of sepal |
| sepal_width | Width of sepal |
| petal_length | Length of petal |
| petal_width | Width of petal |
| species | Iris flower species (target variable) |

The target variable (`species`) contains three classes:

- Setosa
- Versicolor
- Virginica

The dataset is loaded directly from the CSV file into a Spark DataFrame during the notebook execution.

---

# Requirements
The project requires:

- Python 3.x
- Jupyter Notebook
- Apache Spark
- PySpark

Install required libraries:

```bash
pip install pyspark pandas matplotlib
