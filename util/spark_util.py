import uuid
import io

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
    FTSF_LOCATION_FS = "/tmp/delta-tensor-flattened"

    def __init__(self):
        self.spark = get_spark_session()
    
    def clear_cache(self):
        self.spark.catalog.clearCache()

    def write_tensor(self,
                     tensor: np.ndarray | SparseTensorCOO | SparseTensorCSR | SparseTensorCSC | SparseTensorCSF | SparseTensorModeGeneric,
                     is_sparse: bool = False) -> str:
        if is_sparse:
            return self.write_sparse_tensor(tensor)
        return self.write_dense_tensor(tensor)

    def write_dense_tensor(self, tensor: np.ndarray, chunk_dim_chunk: int = 3) -> str:
        tensor_id = str(uuid.uuid4())
        dim_count = tensor.ndim
        dimensions = list(tensor.shape)
        # TODO: strides
        data = [{
            "id": tensor_id,
            "chunk": chunk,
            "chunk_id": i,
            "dim_count": dim_count,
            "chunk_dim_count": chunk_dim_chunk,
            "dimensions": dimensions,
        } for i, chunk in enumerate(self.chunks_binaries(tensor, chunk_dim_chunk))]
        schema = StructType([
            StructField("id", StringType(), False),
            StructField("chunk", BinaryType(), False),
            StructField("chunk_id", IntegerType()),
            StructField("dim_count", IntegerType()),
            StructField("chunk_dim_count", IntegerType()),
            StructField("dimensions", ArrayType(IntegerType())),
        ])

        df = self.spark.createDataFrame(data, schema)
        df.write.format("delta").mode("append").save(SparkUtil.FTSF_LOCATION_FS)
        return tensor_id

    @classmethod
    def flatten_to_chunks(cls, tensor: np.ndarray, chunk_dim_count: int) -> list[np.ndarray]:
        if tensor.ndim <= chunk_dim_count:
            return [tensor]
        
        chunk_dimensions = list(tensor.shape[-chunk_dim_count:])
        flattened_tensor = tensor.reshape([-1] + chunk_dimensions)
        chunks = [flattened_tensor[i] for i in range(flattened_tensor.shape[0])]
        return chunks

    @classmethod
    def chunks_binaries(cls, tensor: np.ndarray, chunk_dim_count: int) -> bytes:
        def get_array_bytes(array: np.ndarray):
            buffer = io.BytesIO()
            np.save(buffer, array)
            return buffer.getvalue()

        chunks = SparkUtil.flatten_to_chunks(tensor, chunk_dim_count)
        chunk_binaries = [get_array_bytes(chunk) for chunk in chunks]
        return chunk_binaries

    def deserialize_from(chunk_rows: list[Row]) -> list[np.ndarray]:
        chunks = []
        for row in chunk_rows:
            buffer = io.BytesIO(row['chunk'])
            chunk = np.load(buffer)
            chunks.append(chunk)
        return chunks

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
        # TODO @evanyfzhou
        # Please include layout as a column in the delta-table
        # df.write.format("delta").mode("append").save("/tmp/delta-tensor-csr")
        raise Exception("Not implemented")

    def __write_csc(self, sparse_tensor: SparseTensorCSC) -> str:
        # TODO @evanyfzhou
        # Please include layout as a column in the delta-table
        # df.write.format("delta").mode("append").save("/tmp/delta-tensor-csc")
        raise Exception("Not implemented")

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
        df = self.spark.read.format("delta").load(SparkUtil.FTSF_LOCATION_FS)
        tensor_df = df.filter(df.id == tensor_id).sort(df.chunk_id.asc())
        chunks = SparkUtil.deserialize_from(tensor_df.select('chunk').collect())
        tensor_shape = tensor_df.select("dimensions").first()['dimensions']
        tensor = np.concatenate(chunks, axis=0).reshape(tensor_shape)
        return tensor

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
        # TODO @evanyfzhou
        raise Exception("Not implemented")

    def __read_csc(self, tensor_id: str) -> SparseTensorCSC:
        # TODO @evanyfzhou
        raise Exception("Not implemented")

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
