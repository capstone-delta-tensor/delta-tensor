import uuid
import io

import pyspark
from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
from pyspark.sql.types import *

from tensor.sparse_tensor import *

MAX_DIGITS = 4 
CHUNK_SIZE = 50000 

def get_spark_session() -> SparkSession:
    builder = pyspark.sql.SparkSession.builder.appName("DeltaTensor") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.driver.memory", "4g")

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

    def split_array(self, arr, chunk_size):
        """Yield successive chunk_size chunks from arr."""
        for i in range(0, len(arr), chunk_size):
            yield arr[i:i + chunk_size]



    def __write_csf(self, sparse_tensor: SparseTensorCSF) -> str:
        # TODO @kevinvan13
        # Please include layout as a column in the delta-table
        # df.write.format("delta").mode("append").save("/tmp/delta-tensor-csf")
        tensor_id = str(uuid.uuid4())


        # Non-chunked data
        fptr_zero = [int(x) for x in sparse_tensor.fptrs[0]]
        fptr_one = [int(x) for x in sparse_tensor.fptrs[1]]
        fid_zero = [int(x) for x in sparse_tensor.fids[0]]
        fid_one = [int(x) for x in sparse_tensor.fids[1]]
        dense_shape = list(sparse_tensor.dense_shape)
        layout = sparse_tensor.layout.name


        # Processing and chunking data with direct conversion to int
        fptr_two_chunks = [[int(x) for x in chunk] for chunk in self.split_array(sparse_tensor.fptrs[2], CHUNK_SIZE)]
        fid_two_chunks = [[int(x) for x in chunk] for chunk in self.split_array(sparse_tensor.fids[2], CHUNK_SIZE)]
        fid_three_chunks = [[int(x) for x in chunk] for chunk in self.split_array(sparse_tensor.fids[3], CHUNK_SIZE)]
        values_chunks = [[float(x) for x in chunk] for chunk in self.split_array(sparse_tensor.values.astype(float).tolist(), CHUNK_SIZE)]


        
        chunked_data = []
        for i in range(max(len(fptr_two_chunks), len(fid_two_chunks), len(fid_three_chunks), len(values_chunks))):
            # Format the chunk index with leading zeros
            padded_index = str(i).zfill(MAX_DIGITS)  # Pads the index with leading zeros
            chunk_id = f"{tensor_id}_{padded_index}"
            chunk_data = {
                "id": chunk_id,
                "tensor_id": tensor_id,
                "layout": layout,
                "dense_shape": dense_shape,
                **({"fptr_two_chunk": fptr_two_chunks[i]} if i < len(fptr_two_chunks) else {}),
                **({"fid_two_chunk": fid_two_chunks[i]} if i < len(fid_two_chunks) else {}),
                **({"fid_three_chunk": fid_three_chunks[i]} if i < len(fid_three_chunks) else {}),
                **({"values_chunk": values_chunks[i]} if i < len(values_chunks) else {}),
                # Including non-chunked data for reference, though could be optimized to store only once
                "fptr_zero": fptr_zero if i == 0 else [],
                "fptr_one": fptr_one if i == 0 else [],
                "fid_zero": fid_zero if i == 0 else [],
                "fid_one": fid_one if i == 0 else [],
            }
            chunked_data.append(chunk_data)
            

        # Define schema including both chunked and non-chunked data
        schema = StructType([
            StructField("id", StringType(), False),
            StructField("tensor_id", StringType(), False),
            StructField("layout", StringType(), False),
            StructField("dense_shape", ArrayType(IntegerType()), False),
            StructField("fptr_zero", ArrayType(IntegerType()), True),
            StructField("fptr_one", ArrayType(IntegerType()), True),
            StructField("fid_zero", ArrayType(IntegerType()), True),
            StructField("fid_one", ArrayType(IntegerType()), True),
            StructField("fptr_two_chunk", ArrayType(IntegerType()), True),
            StructField("fid_two_chunk", ArrayType(IntegerType()), True),
            StructField("fid_three_chunk", ArrayType(IntegerType()), True),
            StructField("values_chunk", ArrayType(DoubleType()), True),
        ])

        df = self.spark.createDataFrame(chunked_data, schema)
        df.write.format("delta").mode("append").save("/tmp/delta-tensor-csf")
        return tensor_id

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
        df = self.spark.read.format("delta").load("/tmp/delta-tensor-csf")
        filtered_df = df.filter(df.tensor_id == tensor_id).sort("id")

        # Initialize empty lists for chunked data
        fptr_two = []
        fid_two = []
        fid_three = []
        values = []

        # Variables for non-chunked data, initially None
        fptr_zero = fptr_one = fid_zero = fid_one = dense_shape = None

        for row in filtered_df.collect():
            # Safely extend lists if column exists and is not None
            if row.asDict().get('fptr_two_chunk') is not None:
                fptr_two.extend(row['fptr_two_chunk'])
            if row.asDict().get('fid_two_chunk') is not None:
                fid_two.extend(row['fid_two_chunk'])
            if row.asDict().get('fid_three_chunk') is not None:
                """
                chunk_size = len(row['fid_three_chunk'])
                print(f"Processing fid_three_chunk with size: {chunk_size}")
                """
                fid_three.extend(row['fid_three_chunk'])
                
            if row.asDict().get('values_chunk') is not None:
                values.extend(row['values_chunk'])

            # Only set non-chunked data if not already set
            if fptr_zero is None:
                fptr_zero = row['fptr_zero'] if 'fptr_zero' in row.asDict() else []
                fptr_one = row['fptr_one'] if 'fptr_one' in row.asDict() else []
                fid_zero = row['fid_zero'] if 'fid_zero' in row.asDict() else []
                fid_one = row['fid_one'] if 'fid_one' in row.asDict() else []
                dense_shape = row['dense_shape'] if 'dense_shape' in row.asDict() else []

        # Ensure dense_shape is correctly formatted if it was ever set
        dense_shape = tuple(dense_shape) if dense_shape is not None else ()

        # Construct and return the SparseTensorCSF object
        fptrs = [fptr_zero or [], fptr_one or [], fptr_two]
        fids = [fid_zero or [], fid_one or [], fid_two, fid_three]
        values = np.array(values) if values else np.array([])

        return SparseTensorCSF(fptrs=fptrs, fids=fids, values=values, dense_shape=dense_shape)

    def __read_mode_generic(self, tensor_id: str) -> SparseTensorModeGeneric:
        df = self.spark.read.format("delta").load("/tmp/delta-tensor-mode-generic")
        filtered_df = df.filter(df.id == tensor_id)
        # filtered_df.show()
        dense_shape, block_shape = filtered_df.select("dense_shape", "block_shape").first()
        indices = np.array(filtered_df.select("index_array").rdd.map(lambda row: row[0]).collect()).transpose()
        values = np.array(filtered_df.select("value").rdd.map(lambda row: row[0]).collect())
        return SparseTensorModeGeneric(indices, values, block_shape, dense_shape)