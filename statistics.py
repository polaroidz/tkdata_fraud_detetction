import pyspark

from pyspark import keyword_only 

from pyspark.sql import functions as F

from pyspark.sql.types import FloatType

from pyspark.ml import Transformer

from pyspark.ml.param.shared import HasInputCol
from pyspark.ml.param.shared import HasOutputCols
from pyspark.ml.param.shared import Param

class Statistics(Transformer, HasInputCol, HasOutputCols):

  @keyword_only
  def __init__(self, inputCol=None, prefix=None, groupByCol=None, statistics=None):
    super(Transformer, self).__init__()

    self.groupByCol = Param(self, "groupByCol", "")
    self.statistics = Param(self, "statistics", "")
    self.prefix     = Param(self, "prefix", "")

    self.setInputCol(inputCol)
    self.setPrefix(prefix)
    self.setGroupByCol(groupByCol)
    self.setStatistics(statistics)

  def setPrefix(self, value):
    return self._set(prefix=value)

  def getPrefix(self):
    return self.getOrDefault(self.prefix)

  def setGroupByCol(self, value):
    return self._set(groupByCol=value)

  def getGroupByCol(self):
    return self.getOrDefault(self.groupByCol)

  def setStatistics(self, value):
    return self._set(statistics=value)

  def getStatistics(self):
    return self.getOrDefault(self.statistics)

  def getOutputCols(self):
    prefix = self.getPrefix()
    stats  = self.getStatistics()

    cols = map(lambda a: "{}_{}".format(prefix, a), stats)
    cols = list(cols)

    return cols

  def _transform(self, df):
    input   = self.getInputCol()
    prefix  = self.getPrefix()
    outputs = self.getOutputCols()
    stats   = self.getStatistics()

    groupByCol = self.getGroupByCol()

    aggs = []

    for stat in stats:
      name = "{}_{}".format(prefix, stat)

      if stat == 'var':
        agg = F.variance(input).alias(name)
      elif stat == 'mean':
        agg = F.mean(input).alias(name)
      elif stat == 'count':
        agg = F.count(input).alias(name)
      elif stat == 'sum':
        agg = F.sum(input).alias(name)
      elif stat == 'nunique' or stat == 'distinct':
        agg = F.countDistinct(input).alias(name)

      aggs.append(agg)

    temp = df.groupBy(groupByCol).agg(*aggs)
    temp = temp.select(*groupByCol, *outputs)

    temp = temp.na.fill(0.0)

    df = df.join(temp, groupByCol, how='left')
    df = df.coalesce(1)

    return df
