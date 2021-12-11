#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


aaaaaaaaaaaaaaaaaaa

import pyspark
import warnings
import pandas as pd
import seaborn as sns
from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder, StandardScaler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import SparkSession

import pyspark
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# In[4]:


spark = SparkSession.builder.master("local").appName("pyspark_giris").getOrCreate()
    
sc = spark.sparkContext


# In[5]:


sc


# In[6]:


spark_df = spark.read.csv("../input/churn/churn.csv", header = True, inferSchema = True)


# In[7]:


spark_df.head()


# # Pandas df vs Spark df

# In[8]:


type(spark_df)


# In[9]:


dff = pd.read_csv('../input/diamonds/diamonds.csv')
type(dff)


# In[10]:


dff.head()


# In[11]:


spark_df.head()


# In[12]:


dff.dtypes


# In[13]:


spark_df.dtypes


# In[14]:


dff.ndim


# In[15]:


# Won't work, every command won't work with spark like dataframe

# spark_df.ndim
# spark_df.shape


# # Exploratory Data Analysis

# In[16]:


# Spark cheatshit needs to be used for all commands


# In[17]:


spark_df.count()


# In[18]:


# Number of observations and variables

spark_df.count(), len(spark_df.columns)


# In[19]:


print("Shape: ", (spark_df.count(), len(spark_df.columns)))


# In[20]:


# types of variables

spark_df.printSchema()


# In[21]:


spark_df.dtypes

spark_df.select('age')


# In[24]:


spark_df.select('age').show(5)


# In[25]:


# head

spark_df.head()


# In[26]:


# take

spark_df.take(5)

spark_df.show(5)

spark_df = spark_df.toDF(*[c.lower() for c in spark_df.columns])


# In[29]:


dff.describe().T


# In[30]:


spark_df.describe().show()


# In[31]:


spark_df


# In[32]:

spark_df.describe('age', 'churn').show()


# In[33]:


spark_df.describe(['age', 'churn']).show()


# In[34]:


# Categorical variable class statistics

# value_counts()

dff['cut'].value_counts()


# In[35]:


spark_df.groupby("churn").count().show()


# In[36]:


# unique

dff['cut'].unique()


# In[37]:


spark_df.select("churn").distinct().show()


# In[38]:


dff.head()


# # len() = .count()

# In[39]:


len(dff['cut'])


# In[40]:


spark_df.filter(spark_df.age > 55).count()


# In[41]:


spark_df.select('age').count()


# # Numeric and categorical variables

# In[42]:


spark_df.show(5)


# In[43]:


num_cols = [i for i in dff.columns if dff[i].dtypes != 'O']
num_cols


# In[44]:


spark_df.columns


# In[45]:


spark_df.dtypes


# In[46]:


num_cols = [i for i in spark_df.columns if spark_df.select(i).dtypes != 'string']
num_cols


# In[47]:


# OR

num_cols = [col[0] for col in spark_df.dtypes if col[1] != 'string']
num_cols


# In[48]:


cat_cols = [i for i in spark_df.columns if spark_df.select(i).dtypes == 'string']
cat_cols


# # toPandas()

# In[49]:


# value_counts() doesn't work with spark
# Converting to pandas

a = spark_df.select('age').toPandas()
a.head()


# In[50]:


a.value_counts().head()


# In[51]:


# len() doesn't work with spark
# Converting to pandas

b = spark_df.select('churn').toPandas()
b.head()


# In[52]:


len(b)


# # Filtering

# In[53]:


dff[dff['carat'] > 4]


# In[54]:


spark_df.filter(spark_df.age > 55).show()


# In[55]:


for col in num_cols:
    spark_df.select(col).distinct().show()


# # groupby

# In[56]:


spark_df.groupby('churn').agg({'age':'mean'}).show()


# # Data Pre-processing & Feature Engineering

# # Missing Values

# In[57]:


dff.isnull().sum()


# In[58]:


from pyspark.sql.functions import when, count, col


# In[59]:


spark_df.select([count(when(col(c).isNull(), c)).alias(c) for c in spark_df.columns]).show()


# In[60]:


c = spark_df.toPandas()
c.head()


# In[61]:


c.isnull().sum()


# In[62]:


spark_df.dropna().show(5)


# In[63]:


spark_df.fillna(50).show(5)


# # Feature Interaction

# In[64]:


spark_df.show(5)


# In[65]:


# If you want to make changes to the variables, use 'withColumn'

spark_df = spark_df.withColumn('age_total_purchase', spark_df.age / spark_df.total_purchase)
spark_df.show(5)


# # Bucketization / Bining / Num to Cat
# 

# In[66]:


from pyspark.ml.feature import Bucketizer


# In[67]:


# In pandas world, this can do with qcut, etc.
# numeric variables to categorical variables

bucketizer = Bucketizer(splits = [0, 30, 45, 65], inputCol = 'age', outputCol = 'age_cat')
spark_df = bucketizer.setHandleInvalid('keep').transform(spark_df)
spark_df.show(15)


# In[68]:


spark_df = spark_df.withColumn('age_cat', spark_df.age_cat + 1)
spark_df.show(5)


# # Creating a variable with when, (segment)

# In[69]:


spark_df = spark_df.withColumn('segment', when(spark_df['years'] < 5, "segment_b").otherwise("segment_a"))
spark_df.show(5)


# In[70]:


spark_df.withColumn('age_cat_2',
                    when(spark_df['age'] < 36, "young").
                    when((35 < spark_df['age']) & (spark_df['age'] < 46), "mature").
                    otherwise("senior")).show(8)


# # cast (double, integer)

# In[71]:


spark_df = spark_df.withColumn("age_cat", spark_df["age_cat"].cast("integer"))


# spark_df = spark_df.withColumn("age_cat", spark_df["age_cat"].cast("double"))
# spark_df.show(5)

# # Label Encoding
# 
# ### Spark like to see dependent variables as feature,
# ### Independent variables as label

# In[72]:


spark_df.show(5)


# In[73]:


indexer = StringIndexer(inputCol="segment", outputCol="segment_label")
temp_sdf = indexer.fit(spark_df).transform(spark_df)
spark_df = temp_sdf.withColumn("segment_label", temp_sdf["segment_label"].cast("integer"))


# In[74]:


spark_df = spark_df.drop('segment')
spark_df.show(15)


# # One Hot Encoding

# In[75]:


encoder = OneHotEncoder(inputCols = ['age_cat'], outputCols = ['age_cat_ohe'])
spark_df = encoder.fit(spark_df).transform(spark_df)
spark_df.show(5)


# # TARGET

# In[76]:


stringIndexer = StringIndexer(inputCol = 'churn', outputCol = 'label')
temp_sdf = stringIndexer.fit(spark_df).transform(spark_df)
spark_df = temp_sdf.withColumn("label", temp_sdf["label"].cast("integer"))


# # Features

# In[77]:


cols = ['age', 'total_purchase', 'account_manager', 'years',
        'num_sites', 'age_total_purchase', 'segment_label', 'age_cat_ohe']


# In[78]:

va = VectorAssembler(inputCols=cols, outputCol="features")
va_df = va.transform(spark_df)
final_df = va_df.select("features", "label")


# In[ ]:


va_df = va.transform(spark_df)
va_df.show()


# In[ ]:


final_df = va_df.select("features", "label")
final_df.show(10)


# # MODELING

# In[ ]:


train_df, test_df = final_df.randomSplit([0.7, 0.3], seed=17)
train_df.show(10)


# In[ ]:


test_df.show(10)


# In[ ]:


print("Training Dataset Count: " + str(train_df.count())), print("Test Dataset Count: " + str(test_df.count()))


# # Logistic Regression

# In[ ]:


log_model = LogisticRegression(featuresCol='features', labelCol='label').fit(train_df)
y_pred = log_model.transform(test_df)
y_pred.show()


# In[ ]:


y_pred.select('label', 'prediction').show()


# In[ ]:


evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", metricName='areaUnderROC')


# In[ ]:


evaluatorMulti = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")


# In[ ]:

acc = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "accuracy"})
roc_auc = evaluator.evaluate(y_pred)
precision = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "precisionByLabel"})
recall = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "recallByLabel"})
f1 = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "f1"})


# In[ ]:


print("accuracy: %f, precision: %f, recall: %f, f1: %f, roc_auc: %f" % (acc, precision, recall, f1, roc_auc))


# # GBM

gbm = GBTClassifier(maxIter=100, featuresCol="features", labelCol="label")
gbm_model = gbm.fit(train_df)
y_pred = gbm_model.transform(test_df)

y_pred.show(5)


# In[ ]:


evaluator = BinaryClassificationEvaluator()

gbm_params = (ParamGridBuilder()
              .addGrid(gbm.maxDepth, [2, 4, 6])
              .addGrid(gbm.maxBins, [20, 30])
              .addGrid(gbm.maxIter, [10, 20])
              .build())

cv = CrossValidator(estimator=gbm,
                    estimatorParamMaps=gbm_params,
                    evaluator=evaluator,
                    numFolds=5)

cv_model = cv.fit(train_df)


# In[ ]:



y_pred = cv_model.transform(test_df)
ac = y_pred.select("label", "prediction")


# # New Prediction
# 
# ### What about new customers come ?

# In[ ]:


names = pd.Series(["John", "jimmy", "Daman", "Quasis", "Uho"])
age = pd.Series([18, 43, 34, 50, 40])
total_purchase = pd.Series([5000, 10000, 6000, 30000, 100000])
account_manager = pd.Series([1, 0, 0, 1, 1])
years = pd.Series([20, 10, 3, 8, 30])
num_sites = pd.Series([2, 8, 8, 6, 50])
age_total_purchase = age / total_purchase
segment_label = pd.Series([1, 1, 0, 1, 1])
age_cat_ohe = pd.Series([1, 1, 0, 2, 1])

new_customers = pd.DataFrame({
    'names': names,
    'age': age,
    'total_purchase': total_purchase,
    'account_manager': account_manager,
    'years': years,
    'num_sites': num_sites,
    "age_total_purchase": age_total_purchase,
    "segment_label": segment_label,
    "age_cat_ohe": age_cat_ohe})


# In[ ]:


new_sdf = spark.createDataFrame(new_customers)
new_customers = va.transform(new_sdf)
new_customers.show(3)


# In[ ]:


results = cv_model.transform(new_customers)
results.select("names", "prediction").show()


# # User Defined Functions (UDFs)

# In[ ]:


from pyspark.sql.types import IntegerType, StringType, FloatType
from pyspark.sql.functions import udf


# In[ ]:


def age_converter(age):
    if age < 35:
        return 1
    elif age < 45:
        return 2
    elif age <= 65:
        return 3


# In[ ]:


func_udf = udf(age_converter, IntegerType())


# In[ ]:


spark_df = spark_df.withColumn('age_cat2', func_udf(spark_df['age']))
spark_df.show(5)


# In[ ]:


def segment(years):
    if years < 5:
        return "segment_b"
    else:
        return "segment_a"


func_udf = udf(segment, StringType())
spark_df = spark_df.withColumn('segment', func_udf(spark_df['years']))
spark_df.show(5)


# # Pandas UDFs

# In[ ]:


from pyspark.sql.functions import pandas_udf


# In[ ]:


@pandas_udf(FloatType())
def pandas_log(col):
    import numpy as np
    return np.log(col)

spark_df.withColumn('age_log', pandas_log(spark_df.age)).show(5)
