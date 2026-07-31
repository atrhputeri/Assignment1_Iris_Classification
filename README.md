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

The dataset used in this project is the iris.csv file provided in the repository. 

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

# How to Run the Project

## Step 1: Clone the Repository

Clone this repository to your local machine:

```bash
git clone <repository-url>
```

Navigate to the project folder:

```bash
cd Iris-Classification-Project
```

---

## Step 2: Open the Jupyter Notebook

Open the notebook file:

```
Iris_Classification_Project.ipynb
```

using Jupyter Notebook.

---

## Step 3: Run the Notebook

Run all notebook cells from beginning to end:

```
Kernel → Restart Kernel and Run All Cells
```

The notebook will automatically perform:

1. Load the Iris dataset
2. Perform data cleaning
3. Prepare data for machine learning
4. Train classification models
5. Tune model parameters
6. Evaluate model performance
7. Compare the three models

---

# Data Preprocessing

## 1. Duplicate Checking

Duplicate records were checked by grouping all columns and counting repeated rows.

Duplicate records were removed using:

```
dropDuplicates()
```

Dataset size after cleaning:

| Dataset | Number of Records |
|---|---:|
| Original Dataset | 150 |
| Cleaned Dataset | 147 |

---

## 2. Missing Value Checking

Missing values were checked for all columns.

Result:

```
No missing values were detected.
```

---

## 3. Label Encoding

The target variable (`species`) was converted into numerical labels because Spark MLlib requires numerical input.

The label mapping is:

| Species | Label |
|---|---:|
| versicolor | 0.0 |
| virginica | 1.0 |
| setosa | 2.0 |

The conversion was performed using **StringIndexer**.

---

## 4. Feature Preparation

The input features were combined into a single feature vector using **VectorAssembler**.

The features used were:

- sepal_length
- sepal_width
- petal_length
- petal_width

---

# Train-Test Split

The cleaned dataset was divided into training and testing datasets.

Split ratio:

- 80% Training data
- 20% Testing data

Result:

| Dataset | Number of Records |
|---|---:|
| Training Data | 123 |
| Testing Data | 24 |

---

# Machine Learning Models

Three classification models from Spark MLlib were implemented.

Hyperparameter tuning was performed using:

- Grid Search
- Cross Validation

---

## Decision Tree Classifier

The Decision Tree model was tuned using:

- maxDepth
- maxBins

Best hyperparameters:

```
maxDepth = 5
maxBins = 32
```

---

## Random Forest Classifier

The Random Forest model was tuned using:

- numTrees
- maxDepth

Best hyperparameters:

```
numTrees = 10
maxDepth = 5
```

---

## Logistic Regression

The Logistic Regression model was tuned using:

- regParam
- elasticNetParam

Best hyperparameters:

```
regParam = 0.01
elasticNetParam = 0.0
```

---

# Model Evaluation

The tuned models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score

The results are shown below:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Decision Tree | 95.83% | 96.43% | 95.83% | 95.70% |
| Random Forest | 95.83% | 96.43% | 95.83% | 95.70% |
| Logistic Regression | 95.83% | 96.43% | 95.83% | 95.70% |

---

# Additional Analysis

## Confusion Matrix Analysis

A confusion matrix was generated to provide a detailed view of the prediction results.

The results showed:

- All Setosa samples were classified correctly.
- All Versicolor samples except one were classified correctly.
- All Virginica samples were classified correctly.
- One Virginica sample was incorrectly predicted as Versicolor.

Overall prediction results:

- Correct predictions: 23 out of 24 samples
- Incorrect predictions: 1 sample

All three models produced the same confusion matrix, showing that they had similar prediction performance.

---

## Misclassification Analysis

The incorrectly classified sample was:

```
Actual class:
Virginica

Predicted class:
Versicolor
```

This error may occur because Virginica and Versicolor have similar characteristics, making them more difficult for the models to distinguish.

---

# Model Comparison

## Decision Tree

### Strengths:
- Easy to understand and interpret
- Provides clear decision rules
- Requires less computational resources

### Limitations:
- Can overfit if the tree becomes too complex

---

## Random Forest

### Strengths:
- More robust because it combines multiple decision trees
- Reduces the risk of overfitting

### Limitations:
- Requires more computational resources
- Less interpretable compared to a single decision tree

---

## Logistic Regression

### Strengths:
- Simple and efficient
- Suitable for classification problems

### Limitations:
- May not capture complex relationships between features

---

# Conclusion

This project successfully implemented a complete machine learning classification workflow using PySpark MLlib.

The Iris dataset was cleaned, transformed, and prepared before applying three classification algorithms: Decision Tree, Random Forest, and Logistic Regression.

After hyperparameter tuning using Grid Search and Cross Validation, all three models achieved the same performance:

- Accuracy: 95.83%
- Precision: 96.43%
- Recall: 95.83%
- F1-score: 95.70%

Since all models produced identical evaluation results and confusion matrices, no single model showed better predictive performance. Therefore, model selection can depend on other factors such as interpretability, robustness, and computational efficiency.

Overall, all three optimized models were able to classify Iris flower species effectively.
