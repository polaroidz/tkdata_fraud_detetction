import math

import pyspark

from pyspark import keyword_only 

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from pyspark.sql.types import LongType

from pyspark.ml import Transformer
from pyspark.ml import Pipeline

from pyspark.ml.feature import VectorAssembler
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

from date import DateColumns
from attribution import AttributionRates

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
dt_assembler = VectorAssembler(inputCols=dt_trans.getOutputColumns(), outputCol="dt_cols", handleInvalid="skip")
dt_minmax = MinMaxScaler(inputCol="dt_cols", outputCol="dt_scaled")

attr_app = AttributionRates(inputCol="is_attributed", groupByCol=["app"], outputCol="attr_app")
attr_device = AttributionRates(inputCol="is_attributed", groupByCol=["device"], outputCol="attr_device")
attr_os = AttributionRates(inputCol="is_attributed", groupByCol=["os"], outputCol="attr_os")
attr_channel = AttributionRates(inputCol="is_attributed", groupByCol=["channel"], outputCol="attr_channel")

attr_app_channel = AttributionRates(inputCol="is_attributed", groupByCol=["app", "channel"], outputCol="attr_app_channel")
attr_app_device = AttributionRates(inputCol="is_attributed", groupByCol=["app", "device"], outputCol="attr_app_device")
attr_app_os = AttributionRates(inputCol="is_attributed", groupByCol=["app", "os"], outputCol="attr_app_os")

dt_pipeline = Pipeline(stages=[
  dt_trans,
  dt_assembler,
  dt_minmax
])

attr_pipeline = Pipeline(stages=[
  attr_app,
  attr_device,
  attr_os,
  attr_channel,
  #attr_app_channel,
  #attr_app_device,
  #attr_app_os
])

pipeline = Pipeline(stages=[
  dt_pipeline,
  attr_pipeline
])

pipeline = pipeline.fit(df)

df = pipeline.transform(df)

df.show()