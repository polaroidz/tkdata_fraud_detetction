import math

import pyspark

from pyspark import keyword_only 

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from pyspark.sql.types import LongType

from pyspark.ml import Transformer
from pyspark.ml import Pipeline

from pyspark.ml.feature import OneHotEncoderEstimator
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import Imputer
from pyspark.ml.feature import PCA

from pyspark.ml.param.shared import HasInputCol
from pyspark.ml.param.shared import HasInputCols
from pyspark.ml.param.shared import HasOutputCol
from pyspark.ml.param.shared import Param
from pyspark.ml.param.shared import Params
from pyspark.ml.param.shared import TypeConverters

from pyspark.ml.classification import LogisticRegression

from date import DateColumns
from conditional import Conditional

# Constants

TARGET = "is_condibuted"

# Arguments

train_path = "/hdfs/tkdata_ads_fraud/raw/train_sample.csv"
#train_path = "/hdfs/tkdata_ads_fraud/raw/train.csv"

output_path  = "/hdfs/tkdata_ads_fraud/data/train_features.parquet"

# Hyperparameters

sql = SparkSession.builder \
  .master("local") \
  .appName("train_pipeline") \
  .getOrCreate()

df = sql.read \
 .format("csv") \
 .option("sep", ",") \
 .option("inferSchema", "true") \
 .option("header", "true") \
 .load(train_path)

positive = df.where(df[TARGET] == 1)
negative = df.where(df[TARGET] == 0)

# Downsapling
fraction = positive.count() / negative.count()
negative = negative.sample(withReplacement=True, fraction=fraction, seed=42)

# Upsampling
#fraction = negative.count() / positive.count()
#positive = positive.sample(withReplacement=True, fraction=fraction, seed=42)

df = positive.union(negative)
df = df.orderBy(F.rand())
df = df.coalesce(4)

dt_trans = DateColumns(inputCol="click_time")
dt_ass = VectorAssembler(inputCols=dt_trans.getOutputColumns(), outputCol="dt_cols", handleInvalid="skip")
dt_minmax = MinMaxScaler(inputCol="dt_cols", outputCol="dt_scaled")

dt_pipeline = Pipeline(stages=[
  dt_trans,
  dt_ass,
  dt_minmax
])

cond_cols = [
  "cond_app",
  "cond_device",
  "cond_os",
  "cond_channel"
]

cond_app = Conditional(inputCol=TARGET, groupByCol=["app"], outputCol="cond_app")
cond_device = Conditional(inputCol=TARGET, groupByCol=["device"], outputCol="cond_device")
cond_os = Conditional(inputCol=TARGET, groupByCol=["os"], outputCol="cond_os")
cond_channel = Conditional(inputCol=TARGET, groupByCol=["channel"], outputCol="cond_channel")

cond_ass = VectorAssembler(inputCols=cond_cols, outputCol="cond_cols", handleInvalid="skip")

cond_pipeline = Pipeline(stages=[
  cond_app,
  cond_device,
  cond_os,
  cond_channel,
  cond_ass
])

# Features

fetures_cols = [
  "dt_scaled",
  "cond_cols"
]

features_ass = VectorAssembler(inputCols=fetures_cols, outputCol="features")

# Final Pipeline

pipeline = Pipeline(stages=[
  dt_pipeline,
  cond_pipeline,
  features_ass
])

pipeline = pipeline.fit(df)

df = pipeline.transform(df)

# Model

lr = LogisticRegression(
  featuresCol="features", 
  labelCol=TARGET,
  predictionCol="predictions",
  maxIter=10,
  regParam=0.0,
  elasticNetParam=0.0,
  threshold=0.5
)

lr = lr.fit(df)
df = lr.transform(df)

summary = lr.summary

print("Labels")
print(summary.labels)

print("Accuracy")
print(summary.accuracy)

print("Precision by Label")
print(summary.precisionByLabel)

print("Recall by Label")
print(summary.recallByLabel)

print("False Positve Rate")
print(summary.falsePositiveRateByLabel)

print("True Positive Rate by Label")
print(summary.truePositiveRateByLabel)

print("Area Under ROC")
print(summary.areaUnderROC)
