#!/usr/bin/env python
# coding: utf-8

# In[1]:


# install pyspark
# Python library for working with Apache Spark

get_ipython().system('pip install pyspark')


# In[20]:


# imports SparkSession from PySpark

from pyspark.sql import SparkSession


# In[ ]:


# Iris Classification using Spark MLlib
# load the Iris dataset using SparkSession
# This dataset contains 4 features and 1 target variable (species)


# In[21]:


# show the table 

spark = SparkSession.builder.appName("IrisClassification").getOrCreate()

df = spark.read.csv("iris.csv", header=True, inferSchema=True)

df.show()


# ### Preprocess data 

# In[60]:


# data preprocessing
# imports col function from PySpark
# col is to refer to dataframe column

from pyspark.sql.functions import col


# In[61]:


# find duplicates 
# group all col together and count how many times each row appear
# if more than 1, consider a duplicate

df_duplicates = df.groupBy(df.columns).count().filter(col("count") > 1)
df_duplicates.show()


# In[24]:


# drop the duplicate 
# the cleaned dataset is set as df_clean

df_clean = df.dropDuplicates()
df_clean.show()


# In[25]:


# check again to make sure duplicates have been remove

df_clean.groupBy(df_clean.columns).count().filter("count > 1").show()


# In[26]:


# to count total value such as missing values or sum column
# import function sum

from pyspark.sql.functions import sum


# In[49]:


# to check if theres any null value in the data 

df_clean.select([sum(col(c).isNull().cast("int")).alias(c) for c in df_clean.columns]).show()


# In[50]:


# display dataset structure
# data types

df_clean.printSchema()


# In[29]:


# compare original data and cleaned data

print("Original:", df.count())
print("Cleaned:", df_clean.count())


# In[30]:


# create label 
# convert column (species) to numerical values 
# important because ML in Spark MLlib only works with numerical data
# So the result is stored in new column named 'label'

from pyspark.ml.feature import StringIndexer

indexer = StringIndexer(inputCol="species", outputCol="label")
df_clean = indexer.fit(df_clean).transform(df_clean)


# In[31]:


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

# In[32]:


# splits the dataset into two parts
# training data and testing data
# 80% of the data is used to train the machine learning model
# 20% is used to test its performance

train_data, test_data = df_clean.randomSplit([0.8, 0.2], seed=42)


# In[33]:


# verify split ratio and make sure data is divided properly 

print("Training data:", train_data.count())
print("Testing data:", test_data.count())

1. Decision Tree
# In[34]:


# apply decision tree model
# created using feature as input and label as target output 
# The model is trained using train_data
# after training, transform() is used to generate predictions on the test dataset

from pyspark.ml.classification import DecisionTreeClassifier

dt = DecisionTreeClassifier(featuresCol="features", labelCol="label")

dt_model = dt.fit(train_data)
dt_predictions = dt_model.transform(test_data)

2. Random Forest
# In[62]:


# import Random Forest classifier
from pyspark.ml.classification import RandomForestClassifier

# create Random Forest model (20 trees)
rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=20)

# train the model using training data
rf_model = rf.fit(train_data)

# make predictions on test data
rf_predictions = rf_model.transform(test_data)

3. Logistic Regression
# In[63]:


# import Logistic Regression classifier
from pyspark.ml.classification import LogisticRegression

# create Logistic Regression model
lr = LogisticRegression(featuresCol="features", labelCol="label")

# train the model using training data
lr_model = lr.fit(train_data)

# make predictions on test data
lr_predictions = lr_model.transform(test_data)

model tuning = trying different settings to get better accuracy.
methods : cross validation & grid search 
# In[40]:


# import tools 

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier


# In[41]:


# import and create Random Forest classifier
# featuresCol = input features used for prediction
# labelCol = target output (species class)

rf = RandomForestClassifier(featuresCol="features", labelCol="label")


# In[42]:


# now test using grid search 

paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [10, 20]) \
    .addGrid(rf.maxDepth, [5, 10]) \
    .build()


# In[64]:


# import evaluation metric for classification
# used to measure model performance (accuracy, F1, etc.)

evaluator = MulticlassClassificationEvaluator(
    labelCol="label",            # actual values
    predictionCol="prediction"   # predicted values
)

# CrossValidator performs model tuning using cross-validation (CV)
# estimator = model to be trained (Random Forest)
# estimatorParamMaps = grid search parameters
# evaluator = performance measurement tool
# numFolds = number of splits for cross-validation

crossval = CrossValidator(
    estimator=rf,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    numFolds=3
)


# In[65]:


# train the model using CV
# this will test different combinations
# and select the best performing model

cv_model = crossval.fit(train_data)


# In[66]:


# generate predictions using the best tuned model
# applies the trained CV model on test data

predictions = cv_model.transform(test_data)


# In[54]:


# Calc accuracy of the model 

accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)
print("Accuracy:", accuracy)


# In[55]:


# calculate precision of the model
# means how many predicted values are actually correct

precision = evaluator.setMetricName("weightedPrecision").evaluate(predictions)
print("Precision:", precision)


# In[56]:


# Recall
# = how many actual correct values were successfully predicted

recall = evaluator.setMetricName("weightedRecall").evaluate(predictions)
print("Recall:", recall)


# In[57]:


# F1 score
# balance between precision and recall
# it shows overall model performance in one value

f1 = evaluator.setMetricName("f1").evaluate(predictions)
print("F1 Score:", f1)


# In[67]:


# function to evaluate a model's performance

def evaluate_model(pred, name):
    print("\n", name)
    print("Accuracy:", evaluator.setMetricName("accuracy").evaluate(pred))
    print("Precision:", evaluator.setMetricName("weightedPrecision").evaluate(pred))
    print("Recall:", evaluator.setMetricName("weightedRecall").evaluate(pred))
    print("F1:", evaluator.setMetricName("f1").evaluate(pred))


# In[69]:


# compares Decision Tree, Random Forest, Logistic Regression,

evaluate_model(dt_predictions, "Decision Tree")
evaluate_model(rf_predictions, "Random Forest")
evaluate_model(lr_predictions, "Logistic Regression")
evaluate_model(predictions, "Tuned Random Forest")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




