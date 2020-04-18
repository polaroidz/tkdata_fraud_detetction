import math

import pyspark

from pyspark import keyword_only 

from pyspark.sql import functions as F

from pyspark.sql.types import FloatType

from pyspark.ml import Transformer

from pyspark.ml.param.shared import HasInputCol
from pyspark.ml.param.shared import HasInputCols
from pyspark.ml.param.shared import HasOutputCol
from pyspark.ml.param.shared import Param
from pyspark.ml.param.shared import Params
from pyspark.ml.param.shared import TypeConverters

class Conditional(Transformer, HasInputCol, HasOutputCol):

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

    udf = F.udf(calculation, FloatType())

    temp = df.groupBy(groupByCol).agg(
      F.count(input).alias("count"),
      F.sum(input).alias("sum")
    )

    temp = temp.withColumn(output, udf("count", "sum"))
    temp = temp.select(*groupByCol, output)

    df = df.join(temp, groupByCol, how='left')
    df = df.coalesce(1)

    return df
