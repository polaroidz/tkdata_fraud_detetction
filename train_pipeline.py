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

class DateColumns(Transformer, HasInputCol):

  @keyword_only
  def __init__(self, inputCol=None):
    super(Transformer, self).__init__()

    self.setInputCol(inputCol)

  def _transform(self, df):
    input = self.getInputCol()

    df = df.withColumn("dt_day", F.dayofmonth(input))
    df = df.withColumn("dt_hour", F.hour(input))
    df = df.withColumn("dt_minute", F.minute(input))
    df = df.withColumn("dt_second", F.second(input))

    df = df.withColumn("dt_dayofyear", F.dayofyear(input))
    df = df.withColumn("dt_dayofweek", F.dayofweek(input))
    df = df.withColumn("dt_weekofyear", F.weekofyear(input))

    return df

  def getOutputColumns(self):
    return [
      "dt_day", 
      "dt_hour", 
      "dt_minute", 
      "dt_second",
      "dt_dayofyear",
      "dt_dayofweek",
      "dt_weekofyear"
    ]

class AttributionRates(Transformer, HasInputCol, HasOutputCol):

  @keyword_only
  def __init__(self, inputCol=None, outputCol=None, groupByCol=None):
    super(Transformer, self).__init__()

    self.groupByCol = Param(self, "groupByCol", "") 

    self.setInputCol(inputCol)
    self.setOutputCol(outputCol)
    self.setGroupByCol(groupByCol)

  def setGroupByCol(self, value):
    return self._set(groupByCol=value)

  def getGroupByCol(self):
    return self.getOrDefault(self.groupByCol)

  def _transform(self, df):
    input  = self.getInputCol()
    output = self.getOutputCol()

    groupByCol = self.getGroupByCol()

    log_group = math.log(100000)

    def calculation(count, _sum):
      rate = _sum / count
      log  = math.log(count) / log_group
      conf = min(1, log)

      return rate * conf

    udf = F.udf(calculation, LongType())

    temp = df.groupBy(groupByCol).agg(
      F.count(input).alias("count"),
      F.sum(input).alias("sum")
    )

    temp = temp.withColumn(output, udf("count", "sum"))
    temp = temp.select(*groupByCol, output)

    df = df.join(temp, groupByCol, how='left')
    df = df.coalesce(1)

    return df


dt_trans = DateColumns(inputCol="click_time")
dt_assembler = VectorAssembler(inputCols=dt_trans.getOutputColumns(), outputCol="dt_cols", handleInvalid="skip")
dt_minmax = MinMaxScaler(inputCol="dt_cols", outputCol="dt_scaled")

attr_app = AttributionRates(inputCol="is_attributed", groupByCol=["app"], outputCol="attr_app")
attr_device = AttributionRates(inputCol="is_attributed", groupByCol=["device"], outputCol="attr_device")
attr_os = AttributionRates(inputCol="is_attributed", groupByCol=["os"], outputCol="attr_os")
attr_channel = AttributionRates(inputCol="is_attributed", groupByCol=["channel"], outputCol="attr_channel")

pipeline = Pipeline(stages=[
#  dt_trans,
#  dt_assembler,
#  dt_minmax,
  attr_app,
  attr_device,
  attr_os,
  attr_channel
])

pipeline = pipeline.fit(df)

df = pipeline.transform(df)

df.show()