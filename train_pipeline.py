import math

import pyspark

from pyspark import keyword_only 

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from pyspark.sql.types import FloatType

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
from pyspark.ml.param.shared import HasOutputCols
from pyspark.ml.param.shared import Param
from pyspark.ml.param.shared import Params
from pyspark.ml.param.shared import TypeConverters

from pyspark.ml.classification import LogisticRegression

from date import DateColumns
from conditional import Conditional
from statistics import Statistics

# Constants

TARGET = "is_attributed"

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

# Datetime

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

# Statistics

st_features  = []

st_app = Statistics(
 inputCol="channel", 
 prefix="app", 
 groupByCol=["ip", "app"], 
 statistics=["count"]
)

st_features += st_app.getOutputCols()

st_channel_device = Statistics(
  inputCol="dt_day", 
  prefix="channel_device", 
  groupByCol=["ip", "app", "channel"], 
  statistics=["var", "count", "mean"]
)

st_features += st_channel_device.getOutputCols()

st_channel_app = Statistics(
  inputCol="channel", 
  prefix="channel_app", 
  groupByCol=["app"], 
  statistics=["count", "distinct"]
)

st_features += st_channel_app.getOutputCols()

st_app_os_hour = Statistics(
 inputCol="dt_hour", 
 prefix="app_os_hour", 
 groupByCol=["ip", "app", "os"], 
 statistics=["var", "count", "mean"]
)

st_features += st_app_os_hour.getOutputCols()

st_app_channel = Statistics(
  inputCol="app", 
  prefix="app_channel", 
  groupByCol=["channel"], 
  statistics=["count", "distinct"]
)

st_features += st_app_channel.getOutputCols()

st_app_day_hour = Statistics(
 inputCol="dt_hour", 
 prefix="app_day_hour", 
 groupByCol=["ip", "app", "dt_day", "dt_hour"], 
 statistics=["mean"]
)

st_features += st_app_day_hour.getOutputCols()

st_ip_channel = Statistics(
  inputCol="channel", 
  prefix="ip_channel", 
  groupByCol=["ip"], 
  statistics=["distinct"]
)

st_features += st_ip_channel.getOutputCols()

st_ip_device = Statistics(
  inputCol="device", 
  prefix="ip_device", 
  groupByCol=["ip"], 
  statistics=["distinct"]
)

st_features += st_ip_device.getOutputCols()

st_ip_device_os_app = Statistics(
  inputCol="app", 
  prefix="ip_device_os_app", 
  groupByCol=["ip", "device", "os"], 
  statistics=["distinct"]
)

st_features += st_ip_device_os_app.getOutputCols()

st_app_os = Statistics(
  inputCol="os", 
  prefix="app_os", 
  groupByCol=["ip", "app"], 
  statistics=["distinct"]
)

st_features += st_app_os.getOutputCols()

st_ip_app = Statistics(
  inputCol="app", 
  prefix="ip_app", 
  groupByCol=["ip"], 
  statistics=["distinct"]
)

st_features += st_ip_app.getOutputCols()

st_ip_day = Statistics(
  inputCol="dt_hour", 
  prefix="ip_day", 
  groupByCol=["ip", "dt_day"], 
  statistics=["distinct"]
)

st_features += st_ip_day.getOutputCols()

st_ass = VectorAssembler(inputCols=st_features, outputCol="st_cols")
st_minmax = MinMaxScaler(inputCol="st_cols", outputCol="st_scaled")

stats_pipeline = Pipeline(stages=[
  st_ip_app,
  st_ip_day,
  st_ip_channel,
  st_ip_device,
  st_ip_device_os_app,
  st_app,
  st_app_channel,
  st_app_os,
  st_app_os_hour,
  st_app_day_hour,
  st_channel_app,
  st_channel_device,
  st_ass,
  st_minmax
])

# Features

features_cols = [
  "dt_scaled", 
  "cond_cols",
  "st_scaled"
]

features_ass = VectorAssembler(inputCols=features_cols, outputCol="features")

# Final Pipeline

pipeline = Pipeline(stages=[
  dt_pipeline,
  cond_pipeline,
  stats_pipeline,
  features_ass
])

# Dataset Transformations

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

# Fitting Pipeline

cols = df.columns

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
