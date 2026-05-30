#!/usr/bin/env python
# coding: utf-8
# The Iris dataset contains 150 flower samples belonging to three species:
# Setosa, Versicolor, and Virginica.

# Each sample contains four numerical measurements:
# sepal length, sepal width, petal length, and petal width.

# The objective is to classify the species based on these measurements.
# In[4]:


# install pyspark
# Python library for working with Apache Spark

get_ipython().system('pip install pyspark')


# In[1]:


# imports SparkSession from PySpark

from pyspark.sql import SparkSession


# In[4]:


# Iris Classification using Spark MLlib
# load the Iris dataset
# the dataset contains 4 features (sepal length, sepal width, petal length and petal width) and 1 target (species)


# In[3]:


# show the table 

spark = SparkSession.builder.appName("IrisClassification").getOrCreate()

df = spark.read.csv("iris.csv", header=True, inferSchema=True)

df.show()


# ### Preprocess data 

# In[4]:


# data preprocessing
# imports col function from PySpark
# col is to refer to dataframe column
# duplicate records were removed and missing values were checked.

from pyspark.sql.functions import col


# In[5]:


# find duplicates 
# group all col together and count how many times each row appear
# if more than 1, consider a duplicate

df_duplicates = df.groupBy(df.columns).count().filter(col("count") > 1)
df_duplicates.show()


# In[6]:


# drop the duplicate 
# the cleaned dataset is set as df_clean

df_clean = df.dropDuplicates()
df_clean.show()


# In[7]:


# check again to make sure duplicates have been remove

df_clean.groupBy(df_clean.columns).count().filter("count > 1").show()


# In[8]:


# to count total value such as missing values or sum column
# import function sum

from pyspark.sql.functions import sum


# In[9]:


# to check if theres any null value in the data 

df_clean.select([sum(col(c).isNull().cast("int")).alias(c) for c in df_clean.columns]).show()


# In[10]:


# compare original data and cleaned data

print("Original:", df.count())
print("Cleaned:", df_clean.count())


# In[11]:


# create label 
# convert column (species) to numerical values 
# important because ML in Spark MLlib only works with numerical data
# So the result is stored in new column named 'label'

from pyspark.ml.feature import StringIndexer

indexer = StringIndexer(inputCol="species", outputCol="label")
df_clean = indexer.fit(df_clean).transform(df_clean)


# In[12]:


# create features
# combine input column into one feature vector
# take numeric columns (sepal length, width etc) and merge into 1 column named 'features'
# this is important because Spark can process all input all together

from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(
    inputCols=["sepal_length", "sepal_width", "petal_length", "petal_width"],
    outputCol="features"
)

df_clean = assembler.transform(df_clean)


# #### split dataset

# In[13]:


# splits the dataset into two parts
# training data and testing data
# 80% of the data is used to train the machine learning model
# 20% is used to test its performance

train_data, test_data = df_clean.randomSplit([0.8, 0.2], seed=42)


# In[14]:


# verify split ratio and make sure data is divided properly 

print("Training data:", train_data.count())
print("Testing data:", test_data.count())


# In[28]:


# import classification algorithms from Spark MLlib
# DecisionTreeClassifier: builds a tree-based model for classification
# RandomForestClassifier: ensemble method using multiple decision trees
# LogisticRegression: linear model used for classification problems

from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier, LogisticRegression

# Import tools for model tuning
# CrossValidator: performs cross-validation to evaluate model performance
# ParamGridBuilder: creates a grid of hyperparameters for tuning

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Import evaluation metric tools
# MulticlassClassificationEvaluator: evaluates classification models using metrics
# such as accuracy, precision, recall, and F1-score

from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# In[29]:


# Create evaluator 
# labelCol is actual value
# predictionCol is model output
# metric used is accuracy
# this will check how correct the prediction is

evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)


# In[30]:


# define models used for classification
models = {
    "Decision Tree": DecisionTreeClassifier(featuresCol="features", labelCol="label"),
    "Random Forest": RandomForestClassifier(featuresCol="features", labelCol="label"),
    "Logistic Regression": LogisticRegression(featuresCol="features", labelCol="label")
}

# define hyperparameter grid for each model
# this is used for tuning the model to get better performance
paramGrids = {
    "Decision Tree": ParamGridBuilder()
        .addGrid(models["Decision Tree"].maxDepth, [3, 5])
        .addGrid(models["Decision Tree"].maxBins, [16, 32])
        .build(),

    "Random Forest": ParamGridBuilder()
        .addGrid(models["Random Forest"].numTrees, [10, 20])
        .addGrid(models["Random Forest"].maxDepth, [5, 10])
        .build(),

    "Logistic Regression": ParamGridBuilder()
        .addGrid(models["Logistic Regression"].regParam, [0.01, 0.1])
        .addGrid(models["Logistic Regression"].elasticNetParam, [0.0, 0.5])
        .build()
}


# In[31]:


# Training and tuning


# store results for all models
results = {}

# loop through each model (Decision Tree, Random Forest, Logistic Regression)
for name in models:

    print("\n====================")
    print(name)
    print("====================")


    # apply cross validation with hyperparameter tuning
    # this will try different parameter combinations from paramGrids
    crossval = CrossValidator(
        estimator=models[name],
        estimatorParamMaps=paramGrids[name],
        evaluator=evaluator,
        numFolds=3
    )

    # train model using training data
    cv_model = crossval.fit(train_data)

    # test model using test data
    predictions = cv_model.transform(test_data)

    # evaluate model performance
    accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)
    precision = evaluator.setMetricName("weightedPrecision").evaluate(predictions)
    recall = evaluator.setMetricName("weightedRecall").evaluate(predictions)
    f1 = evaluator.setMetricName("f1").evaluate(predictions)

    # store results for comparison later
    results[name] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    # print results for each model
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)

# 1. The Decision Tree model achieved an accuracy of 95.83%, which shows that it performs well in classifying the Iris dataset. However, there are still a small number of misclassifications, indicating that a single decision tree may not fully capture all patterns in the data.
# 2. The Random Forest model achieved 100% accuracy, precision, recall, and F1-score, which indicates perfect classification on this dataset. This shows that ensemble learning improves performance by combining multiple decision trees and reducing errors. However, such perfect results may also suggest that the Iris dataset is very simple or easily separable.
# 3. Logistic Regression achieved 95.83% accuracy, which is similar to the Decision Tree model. This indicates that linear classification works reasonably well for this dataset. However, it performs slightly worse than Random Forest, suggesting that non-linear relationships in the data are better captured by tree-based models.
# In[32]:


# compare result

# import pandas library for creating table format
import pandas as pd

# convert results dictionary into a dataframe
# .T is used to transpose
results_df = pd.DataFrame(results).T

# display final comparison table of all models
print(results_df)

#The table above shows the performance of the three classification models evaluated using accuracy, precision, recall, and F1-score.
#In conclusion, Random Forest is the best model for this dataset because it achieved perfect scores in all evaluation metrics. Decision Tree and Logistic Regression performed similarly but slightly lower. This suggests that ensemble learning methods are more suitable for this classification problem. 
#The results also indicate that the Iris dataset is well-structured and easily separable, which allows high performance from classical machine learning models.
