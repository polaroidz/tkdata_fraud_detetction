import math

import pyspark

from pyspark import keyword_only 

from pyspark.sql import functions as F

from pyspark.ml import Transformer

from pyspark.ml.param.shared import HasInputCol
from pyspark.ml.param.shared import HasInputCols
from pyspark.ml.param.shared import HasOutputCol
from pyspark.ml.param.shared import Param
from pyspark.ml.param.shared import Params
from pyspark.ml.param.shared import TypeConverters

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