import uuid

import pyspark
from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
from pyspark.sql.types import *

from tensor.sparse_tensor import *


def get_spark_session() -> SparkSession:
    builder = pyspark.sql.SparkSession.builder.appName("DeltaTensor") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.driver.memory", "16g")

    return configure_spark_with_delta_pip(builder).getOrCreate()


class SparkUtil:
    def __init__(self):
        self.spark = get_spark_session()

    def write_tensor(self,
                     tensor: np.ndarray | SparseTensorCOO | SparseTensorCSR | SparseTensorCSC | SparseTensorCSF | SparseTensorModeGeneric,
                     is_sparse: bool = False) -> str:
        if is_sparse:
            return self.write_sparse_tensor(tensor)
        return self.write_dense_tensor(tensor)

    def write_dense_tensor(self, tensor: np.ndarray) -> str:
        # TODO @LiaoliaoLiu
        raise Exception("Not implemented")

    def write_sparse_tensor(self,
                            tensor: SparseTensorCOO | SparseTensorCSR | SparseTensorCSC | SparseTensorCSF | SparseTensorModeGeneric) -> str:
        match tensor.layout:
            case SparseTensorLayout.COO:
                return self.__write_coo(tensor)
            case SparseTensorLayout.CSR:
                return self.__write_csr(tensor)
            case SparseTensorLayout.CSC:
                return self.__write_csc(tensor)
            case SparseTensorLayout.CSF:
                return self.__write_csf(tensor)
            case SparseTensorLayout.MODE_GENERIC:
                return self.__write_mode_generic(tensor)
            case _:
                raise Exception(f"Layout {tensor.layout} not supported")

    def __write_coo(self, sparse_tensor: SparseTensorCOO) -> str:
        # TODO @920fandanny
        # Please include layout as a column in the delta-table
        # df.write.format("delta").mode("append").save("/tmp/delta-tensor-coo")
        raise Exception("Not implemented")

    def __write_csr(self, sparse_tensor: SparseTensorCSR) -> str:
        tensor_id = str(uuid.uuid4())
        crow_indices = sparse_tensor.crow_indices.tolist()
        col_indices = sparse_tensor.col_indices.tolist()
        values = sparse_tensor.values.tolist()
        dense_shape = sparse_tensor.dense_shape
        layout = sparse_tensor.layout.name
        data = {
            "id": tensor_id,
            "layout": layout,
            "dense_shape": list(dense_shape),
            "crow_indices": crow_indices,
            "col_indices": col_indices,
            "value": values,
        }
        schema = StructType([
            StructField("id", StringType(), False),
            StructField("layout", StringType(), False),
            StructField("dense_shape", ArrayType(IntegerType())),
            StructField("crow_indices", ArrayType(IntegerType())),
            StructField("col_indices", ArrayType(IntegerType())),
            StructField("value", ArrayType(DoubleType())),
        ])
        df = self.spark.createDataFrame([data], schema)
        df.write.format("delta").mode("append").save("/tmp/delta-tensor-csr")
        return tensor_id
    
    def __write_csc(self, sparse_tensor: SparseTensorCSC) -> str:
        tensor_id = str(uuid.uuid4())
        ccol_indices = sparse_tensor.ccol_indices.tolist()
        row_indices = sparse_tensor.row_indices.tolist()
        values = sparse_tensor.values.tolist()
        dense_shape = sparse_tensor.dense_shape
        layout = sparse_tensor.layout.name
        data = {
            "id": tensor_id,
            "layout": layout,
            "dense_shape": list(dense_shape),
            "ccol_indices": ccol_indices,
            "row_indices": row_indices,
            "value": values,
        }
        schema = StructType([
            StructField("id", StringType(), False),
            StructField("layout", StringType(), False),
            StructField("dense_shape", ArrayType(IntegerType())),
            StructField("ccol_indices", ArrayType(IntegerType())),
            StructField("row_indices", ArrayType(IntegerType())),
            StructField("value", ArrayType(DoubleType())),
        ])
        df = self.spark.createDataFrame([data], schema)
        df.write.format("delta").mode("append").save("/tmp/delta-tensor-csc")
        return tensor_id

    def __write_csf(self, sparse_tensor: SparseTensorCSF) -> str:
        # TODO @kevinvan13
        # Please include layout as a column in the delta-table
        # df.write.format("delta").mode("append").save("/tmp/delta-tensor-csf")
        raise Exception("Not implemented")

    def __write_mode_generic(self, sparse_tensor: SparseTensorModeGeneric) -> str:
        tensor_id = str(uuid.uuid4())
        indices = sparse_tensor.indices
        values = sparse_tensor.values
        block_shape = sparse_tensor.block_shape
        dense_shape = sparse_tensor.dense_shape
        layout = sparse_tensor.layout
        data = [{
            "id": tensor_id,
            "layout": layout.name,
            "dense_shape": dense_shape,
            "block_shape": block_shape,
            "index_array": indices[:, i].tolist(),
            "value": values[i].tolist(),
        } for i in range(len(values))]
        schema = StructType([
            StructField("id", StringType(), False),
            StructField("layout", StringType(), False),
            StructField("dense_shape", ArrayType(IntegerType())),
            StructField("block_shape", ArrayType(IntegerType())),
            StructField("index_array", ArrayType(IntegerType())),
            StructField("value", ArrayType(DoubleType())),
        ])

        df = self.spark.createDataFrame(data, schema)
        # df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save("/tmp/delta-tensor-mode-generic")
        df.write.format("delta").mode("append").save("/tmp/delta-tensor-mode-generic")
        return tensor_id

    def read_tensor(self, tensor_id: str, is_sparse: bool = False,
                    layout: SparseTensorLayout = SparseTensorLayout.MODE_GENERIC) -> np.ndarray | SparseTensorCOO | SparseTensorCSR | SparseTensorCSC | SparseTensorCSF | SparseTensorModeGeneric:
        if is_sparse:
            return self.read_sparse_tensor(tensor_id, layout)
        return self.read_dense_tensor(tensor_id)

    def read_dense_tensor(self, tensor_id: str) -> np.ndarray:
        # TODO @LiaoliaoLiu
        raise Exception("Not implemented")

    def read_sparse_tensor(self, tensor_id: str,
                           layout: SparseTensorLayout) -> SparseTensorCOO | SparseTensorCSR | SparseTensorCSC | SparseTensorCSF | SparseTensorModeGeneric:
        match layout:
            case SparseTensorLayout.COO:
                return self.__read_coo(tensor_id)
            case SparseTensorLayout.CSR:
                return self.__read_csr(tensor_id)
            case SparseTensorLayout.CSC:
                return self.__read_csc(tensor_id)
            case SparseTensorLayout.CSF:
                return self.__read_csf(tensor_id)
            case SparseTensorLayout.MODE_GENERIC:
                return self.__read_mode_generic(tensor_id)
            case _:
                raise Exception(f"Layout {layout} not supported")

    def __read_coo(self, tensor_id: str) -> SparseTensorCOO:
        # TODO @920fandanny
        raise Exception("Not implemented")

    def __read_csr(self, tensor_id: str) -> SparseTensorCSR:
        df = self.spark.read.format("delta").load("/tmp/delta-tensor-csr")
        filtered_df = df.filter(df.id == tensor_id)
        dense_shape = filtered_df.select("dense_shape").first()[0]
        crow_indices = np.array(filtered_df.select("crow_indices").rdd.map(lambda row: row[0]).collect())[0]
        col_indices = np.array(filtered_df.select("col_indices").rdd.map(lambda row: row[0]).collect())[0]
        values = np.array(filtered_df.select("value").rdd.map(lambda row: row[0]).collect())[0]
        return SparseTensorCSR(values, col_indices, crow_indices, dense_shape)

    def __read_csc(self, tensor_id: str) -> SparseTensorCSC:
        df = self.spark.read.format("delta").load("/tmp/delta-tensor-csc")
        filtered_df = df.filter(df.id == tensor_id)
        dense_shape = filtered_df.select("dense_shape").first()[0]
        ccol_indices = np.array(filtered_df.select("ccol_indices").rdd.map(lambda row: row[0]).collect())[0]
        row_indices = np.array(filtered_df.select("row_indices").rdd.map(lambda row: row[0]).collect())[0]
        values = np.array(filtered_df.select("value").rdd.map(lambda row: row[0]).collect())[0]
        return SparseTensorCSC(values, row_indices, ccol_indices, dense_shape)

    def __read_csf(self, tensor_id: str) -> SparseTensorCSF:
        # TODO @kevinvan13
        raise Exception("Not implemented")

    def __read_mode_generic(self, tensor_id: str) -> SparseTensorModeGeneric:
        df = self.spark.read.format("delta").load("/tmp/delta-tensor-mode-generic")
        filtered_df = df.filter(df.id == tensor_id)
        # filtered_df.show()
        dense_shape, block_shape = filtered_df.select("dense_shape", "block_shape").first()
        indices = np.array(filtered_df.select("index_array").rdd.map(lambda row: row[0]).collect()).transpose()
        values = np.array(filtered_df.select("value").rdd.map(lambda row: row[0]).collect())
        return SparseTensorModeGeneric(indices, values, block_shape, dense_shape)