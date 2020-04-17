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
from attribution import AttributionRates

# Constants

TARGET = "is_attributed"

# Arguments

train_path = "/hdfs/tkdata_ads_fraud/raw/train_sample.csv"
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

dt_trans = DateColumns(inputCol="click_time")
dt_ass = VectorAssembler(inputCols=dt_trans.getOutputColumns(), outputCol="dt_cols", handleInvalid="skip")
dt_minmax = MinMaxScaler(inputCol="dt_cols", outputCol="dt_scaled")

dt_pipeline = Pipeline(stages=[
  dt_trans,
  dt_ass,
  dt_minmax
])

attr_cols = [
  "attr_app",
  "attr_device",
  "attr_os",
  "attr_channel",
  #"attr_app_channel",
  #"attr_app_device",
  #"attr_app_os"
]

attr_app = AttributionRates(inputCol=TARGET, groupByCol=["app"], outputCol="attr_app")
attr_device = AttributionRates(inputCol=TARGET, groupByCol=["device"], outputCol="attr_device")
attr_os = AttributionRates(inputCol=TARGET, groupByCol=["os"], outputCol="attr_os")
attr_channel = AttributionRates(inputCol=TARGET, groupByCol=["channel"], outputCol="attr_channel")

attr_app_channel = AttributionRates(inputCol=TARGET, groupByCol=["app", "channel"], outputCol="attr_app_channel")
attr_app_device = AttributionRates(inputCol=TARGET, groupByCol=["app", "device"], outputCol="attr_app_device")
attr_app_os = AttributionRates(inputCol=TARGET, groupByCol=["app", "os"], outputCol="attr_app_os")

attr_ass = VectorAssembler(inputCols=attr_cols, outputCol="attr_cols", handleInvalid="skip")

attr_pipeline = Pipeline(stages=[
  attr_app,
  attr_device,
  attr_os,
  attr_channel,
  #attr_app_channel,
  #attr_app_device,
  #attr_app_os,
  attr_ass
])

# Features

fetures_cols = [
  "dt_scaled",
  "attr_cols"
]

features_ass = VectorAssembler(inputCols=fetures_cols, outputCol="features")

# Final Pipeline

pipeline = Pipeline(stages=[
  dt_pipeline,
  attr_pipeline,
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