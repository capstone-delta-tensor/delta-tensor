import uuid

import numpy as np
import pyspark
from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
from pyspark.sql.types import *

from sparse_tensor import SparseTensorModeGeneric


def get_spark_session() -> SparkSession:
    builder = pyspark.sql.SparkSession.builder.appName("DeltaTensor") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")

    return configure_spark_with_delta_pip(builder).getOrCreate()


class SparkUtil:
    def __init__(self):
        self.spark = get_spark_session()

    def write_tensor(self, sparse_tensor: SparseTensorModeGeneric) -> str:
        tensor_id = str(uuid.uuid4())
        indices = sparse_tensor.indices
        values = sparse_tensor.values
        block_shape = sparse_tensor.block_shape
        dense_shape = sparse_tensor.dense_shape
        data = [{
            "id": tensor_id,
            "dense_shape": dense_shape,
            "block_shape": block_shape,
            "index_array": indices[:, i].tolist(),
            "value": values[i].tolist(),
        } for i in range(len(values))]
        schema = StructType([
            StructField("id", StringType(), False),
            StructField("dense_shape", ArrayType(IntegerType())),
            StructField("block_shape", ArrayType(IntegerType())),
            StructField("index_array", ArrayType(IntegerType())),
            StructField("value", ArrayType(DoubleType())),
        ])

        df = self.spark.createDataFrame(data, schema)
        # df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save("/tmp/delta-tensor")
        df.write.format("delta").mode("append").save("/tmp/delta-tensor")
        return tensor_id

    def read_tensor(self, tensor_id: str) -> SparseTensorModeGeneric:
        df = self.spark.read.format("delta").load("/tmp/delta-tensor")
        filtered_df = df.filter(df.id == tensor_id)
        filtered_df.show()
        dense_shape, block_shape = filtered_df.select("dense_shape", "block_shape").first()
        indices = np.array(filtered_df.select("index_array").rdd.map(lambda row: row[0]).collect()).transpose()
        values = np.array(filtered_df.select("value").rdd.map(lambda row: row[0]).collect())
        return SparseTensorModeGeneric(indices, values, block_shape, dense_shape)
