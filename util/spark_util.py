import io
import uuid

import pyspark
from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
from pyspark.sql.types import *

from settings import config
from tensor.sparse_tensor import *

MAX_DIGITS = 4
CHUNK_SIZE = 50000

MAX_DIGITS = 4
CHUNK_SIZE = 50000


def get_spark_session() -> SparkSession:
    builder = pyspark.sql.SparkSession.builder.appName("DeltaTensor") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.driver.maxResultSize", config["spark.driver.maxResultSize"]) \
        .config("spark.driver.memory", config["spark.driver.memory"]) \
        .config("spark.executor.memory", config["spark.executor.memory"]) \
        .config("spark.default.parallelism", config["spark.default.parallelism"]) \
        .config("spark.sql.debug.maxToStringFields", config["spark.sql.debug.maxToStringFields"]) \
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", config["spark.hadoop.fs.s3a.aws.credentials.provider"]) \
        .config("spark.hadoop.fs.s3a.access.key", config["spark.hadoop.fs.s3a.access.key"]) \
        .config("spark.hadoop.fs.s3a.secret.key", config["spark.hadoop.fs.s3a.secret.key"]) \
        .config("spark.hadoop.fs.s3a.session.token", config["spark.hadoop.fs.s3a.session.token"])
    extra_packages = config["extra_packages"]
    return configure_spark_with_delta_pip(builder, extra_packages=extra_packages).getOrCreate()


class SparkUtil:
    BUCKET = config["s3.bucket.name"] if config["s3.bucket.name"] else "/tmp/delta-tensor"
    FTSF_TABLE = '/'.join((BUCKET, "flattened"))
    COO_TABLE = '/'.join((BUCKET, "coo"))
    CSR_TABLE = '/'.join((BUCKET, "csr"))
    CSC_TABLE = '/'.join((BUCKET, "csc"))
    CSF_TABLE = '/'.join((BUCKET, "csf"))
    MODE_GENERIC_TABLE = '/'.join((BUCKET, "mode-generic"))

    def __init__(self):
        self.spark = get_spark_session()


    def stop_session(self):
        self.spark.stop()

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
        df.write.format("delta").mode("append").save(SparkUtil.FTSF_TABLE)
        return tensor_id

    @staticmethod
    def flatten_to_chunks(tensor: np.ndarray, chunk_dim_count: int) -> list[np.ndarray]:
        if tensor.ndim <= chunk_dim_count:
            return [tensor]


        chunk_dimensions = list(tensor.shape[-chunk_dim_count:])
        flattened_tensor = tensor.reshape([-1] + chunk_dimensions)
        chunks = [flattened_tensor[i]
                  for i in range(flattened_tensor.shape[0])]
        return chunks

    @staticmethod
    def get_array_bytes(array: np.ndarray):
        buffer = io.BytesIO()
        np.save(buffer, array)
        return buffer.getvalue()

    @staticmethod
    def chunks_binaries(tensor: np.ndarray, chunk_dim_count: int) -> list[bytes]:
        chunks = SparkUtil.flatten_to_chunks(tensor, chunk_dim_count)
        chunk_binaries = [SparkUtil.get_array_bytes(chunk) for chunk in chunks]
        return chunk_binaries

    @staticmethod
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
        tensor_id = str(uuid.uuid4())
        indices = sparse_tensor.indices
        values = sparse_tensor.values
        layout = sparse_tensor.layout.name
        dense_shape = list(sparse_tensor.dense_shape)
        data = [{
            "id": tensor_id,
            "layout": layout,
            "dense_shape": dense_shape,
            "indices": indices[:, i].tolist(),
            "value": float(values[i]),
        } for i in range(len(values))]
        schema = StructType([
            StructField("id", StringType(), False),
            StructField("layout", StringType(), False),
            StructField("dense_shape", ArrayType(IntegerType())),
            StructField("indices", ArrayType(IntegerType())),
            StructField("value", DoubleType()),
        ])
        df = self.spark.createDataFrame(data, schema)
        df.write.format("delta").mode("append").save(SparkUtil.COO_TABLE)
        return tensor_id

    def __write_csr(self, sparse_tensor: SparseTensorCSR) -> str:
        tensor_id = str(uuid.uuid4())
        crow_indices = sparse_tensor.crow_indices.tolist()
        col_indices = sparse_tensor.col_indices.tolist()
        values = sparse_tensor.values.tolist()
        original_shape = sparse_tensor.original_shape
        dense_shape = sparse_tensor.dense_shape
        layout = sparse_tensor.layout.name
        data = {
            "id": tensor_id,
            "layout": layout,
            "original_shape": list(original_shape),
            "dense_shape": list(dense_shape),
            "crow_indices": crow_indices,
            "col_indices": col_indices,
            "value": values,
        }
        schema = StructType([
            StructField("id", StringType(), False),
            StructField("layout", StringType(), False),
            StructField("original_shape", ArrayType(IntegerType())),
            StructField("dense_shape", ArrayType(IntegerType())),
            StructField("crow_indices", ArrayType(IntegerType())),
            StructField("col_indices", ArrayType(IntegerType())),
            StructField("value", ArrayType(DoubleType())),
        ])
        df = self.spark.createDataFrame([data], schema)
        print("df: ", df)
        df.write.format("delta").mode("append").save(SparkUtil.CSR_TABLE)
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
        df.write.format("delta").mode("append").save(SparkUtil.CSC_TABLE)
        return tensor_id

    def split_array(self, arr, chunk_size):
        """Yield successive chunk_size chunks from arr."""
        for i in range(0, len(arr), chunk_size):
            yield arr[i:i + chunk_size]

    def __write_csf(self, sparse_tensor: SparseTensorCSF) -> str:
        num_dimensions = len(sparse_tensor.fids)
        prefix = "csf"
        dimension_postfix = f"{num_dimensions:02d}"  # Ensure it's two digits
        tensor_id = prefix + str(uuid.uuid4()) + dimension_postfix

        # Non-chunked data
        fptr_zero = [int(x) for x in sparse_tensor.fptrs[0]]
        fptr_one = [int(x) for x in sparse_tensor.fptrs[1]]
        fid_zero = [int(x) for x in sparse_tensor.fids[0]]
        fid_one = [int(x) for x in sparse_tensor.fids[1]]
        dense_shape = list(sparse_tensor.dense_shape)
        layout = sparse_tensor.layout.name

        # Initialize chunked data containers
        chunked_data = []
        values_chunks = [[float(x) for x in chunk] for chunk in
                         self.split_array(sparse_tensor.values.astype(float).tolist(), CHUNK_SIZE)]

        # Dynamically handling dimensions
        fptr_chunks = {}
        fid_chunks = {}
        for i in range(2, len(sparse_tensor.fptrs)):  # Start from the third dimension
            fptr_chunks[f"fptr_{i}_chunk"] = [[int(x) for x in chunk] for chunk in
                                              self.split_array(sparse_tensor.fptrs[i], CHUNK_SIZE)]
        for i in range(2, len(sparse_tensor.fids)):
            fid_chunks[f"fid_{i}_chunk"] = [[int(x) for x in chunk] for chunk in
                                            self.split_array(sparse_tensor.fids[i], CHUNK_SIZE)]

        max_chunks_length = max(len(values_chunks), max((len(chunks) for chunks in fptr_chunks.values()), default=0),
                                max((len(chunks) for chunks in fid_chunks.values()), default=0))

        for i in range(max_chunks_length):
            padded_index = str(i).zfill(MAX_DIGITS)
            chunk_id = f"{tensor_id}_{padded_index}"
            chunk = {
                "id": chunk_id,
                "tensor_id": tensor_id,
                "layout": layout,
                "dense_shape": dense_shape,
                "values_chunk": values_chunks[i] if i < len(values_chunks) else [],
                "fptr_zero": fptr_zero if i == 0 else [],
                "fptr_one": fptr_one if i == 0 else [],
                "fid_zero": fid_zero if i == 0 else [],
                "fid_one": fid_one if i == 0 else [],
            }
            for dim, chunks in fptr_chunks.items():
                if i < len(chunks):
                    chunk[dim] = chunks[i]
            for dim, chunks in fid_chunks.items():
                if i < len(chunks):
                    chunk[dim] = chunks[i]

            chunked_data.append(chunk)

        # Define schema including both chunked and non-chunked data
        fields = [
            StructField("id", StringType(), False),
            StructField("tensor_id", StringType(), False),
            StructField("layout", StringType(), False),
            StructField("dense_shape", ArrayType(IntegerType()), False),
            StructField("fptr_zero", ArrayType(IntegerType()), True),
            StructField("fptr_one", ArrayType(IntegerType()), True),
            StructField("fid_zero", ArrayType(IntegerType()), True),
            StructField("fid_one", ArrayType(IntegerType()), True),
            StructField("values_chunk", ArrayType(DoubleType()), True),
        ]

        # Dynamically adding fields for fptr and fid chunks
        # Dynamically adding fields for fptr and fid chunks
        for dim in range(2, max(len(sparse_tensor.fptrs), len(sparse_tensor.fids))):
            if dim < len(sparse_tensor.fptrs):
                fields.append(StructField(
                    f"fptr_{dim}_chunk", ArrayType(IntegerType()), True))
            if dim < len(sparse_tensor.fids):
                fields.append(StructField(f"fid_{dim}_chunk", ArrayType(IntegerType()), True))

        schema = StructType(fields)
        df = self.spark.createDataFrame(chunked_data, schema)
        path = f"{SparkUtil.CSF_TABLE}/dim_{num_dimensions}/"
        df.write.format("delta").mode("append").save(path)
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
            "value": values[i].astype(float).tolist(),
        } for i in range(len(values))]
        schema = StructType([
            StructField("id", StringType(), False),
            StructField("layout", StringType(), False),
            StructField("dense_shape", ArrayType(IntegerType())),
            StructField("block_shape", ArrayType(IntegerType())),
            StructField("index_array", ArrayType(IntegerType())),
            StructField("value", ArrayType(ArrayType(DoubleType()))),
        ])

        df = self.spark.createDataFrame(data, schema)
        # df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save("/tmp/delta-tensor-mode-generic")
        df.write.format("delta").mode("append").save(SparkUtil.MODE_GENERIC_TABLE)
        return tensor_id

    def read_tensor(self, tensor_id: str, is_sparse: bool = False,
                    layout: SparseTensorLayout = SparseTensorLayout.MODE_GENERIC,
                    slice_tuple: tuple = ()) -> np.ndarray | SparseTensorCOO | SparseTensorCSR | SparseTensorCSC | SparseTensorCSF | SparseTensorModeGeneric:
        if is_sparse:
            return self.read_sparse_tensor(tensor_id, layout, slice_tuple)
        return self.read_dense_tensor(tensor_id, slice_tuple)

    def read_dense_tensor(self, tensor_id: str, slice_tuple: tuple = ()) -> np.ndarray:
        # TODO @LiaoliaoLiu support slicing operation
        df = self.spark.read.format("delta").load(SparkUtil.FTSF_TABLE)
        tensor_df = df.filter(df.id == tensor_id).sort(df.chunk_id.asc())
        chunks = SparkUtil.deserialize_from(
            tensor_df.select('chunk').collect())
        tensor_shape = tensor_df.select("dimensions").first()['dimensions']
        tensor = np.concatenate(chunks, axis=0).reshape(tensor_shape)
        return tensor

    def read_sparse_tensor(self, tensor_id: str,
                           layout: SparseTensorLayout,
                           slice_tuple: tuple = ()) -> SparseTensorCOO | SparseTensorCSR | SparseTensorCSC | SparseTensorCSF | SparseTensorModeGeneric:
        match layout:
            case SparseTensorLayout.COO:
                return self.__read_coo(tensor_id, slice_tuple)
            case SparseTensorLayout.CSR:
                return self.__read_csr(tensor_id, slice_tuple)
            case SparseTensorLayout.CSC:
                return self.__read_csc(tensor_id, slice_tuple)
            case SparseTensorLayout.CSF:
                return self.__read_csf(tensor_id, slice_tuple)
            case SparseTensorLayout.MODE_GENERIC:
                return self.__read_mode_generic(tensor_id, slice_tuple)
            case _:
                raise Exception(f"Layout {layout} not supported")

    def __read_coo(self, tensor_id: str, slice_tuple: tuple) -> SparseTensorCOO:
        # TODO @920fandanny support slicing operation
        df = self.spark.read.format("delta").load(SparkUtil.COO_TABLE)
        filtered_df = df.filter(df.id == tensor_id)
        # print(filtered_df.show())
        selected_data = filtered_df.select(
            "indices", "value", "dense_shape").collect()
        dense_shape = tuple(
            selected_data[0]['dense_shape']) if selected_data else None
        values = [int(row['value']) if row['value'].is_integer()
                  else row['value'] for row in selected_data]
        values = np.array(values)
        indices = [row['indices'] for row in selected_data]
        indices = np.array(indices).transpose()
        tensor = SparseTensorCOO(indices, values, dense_shape)
        return tensor

    def __read_csr(self, tensor_id: str, slice_tuple: tuple) -> SparseTensorCSR:
        # TODO @evanyfzhou support slicing operation
        df = self.spark.read.format("delta").load(SparkUtil.CSR_TABLE)
        filtered_df = df.filter(df.id == tensor_id)
        original_shape = filtered_df.select("original_shape").first()[0]
        dense_shape = filtered_df.select("dense_shape").first()[0]
        crow_indices = np.array(filtered_df.select(
            "crow_indices").rdd.map(lambda row: row[0]).collect())[0]
        col_indices = np.array(filtered_df.select(
            "col_indices").rdd.map(lambda row: row[0]).collect())[0]
        values = np.array(filtered_df.select(
            "value").rdd.map(lambda row: row[0]).collect())[0]
        return SparseTensorCSR(values, col_indices, crow_indices, dense_shape)
        crow_indices = np.array(filtered_df.select("crow_indices").rdd.map(lambda row: row[0]).collect())[0]
        col_indices = np.array(filtered_df.select("col_indices").rdd.map(lambda row: row[0]).collect())[0]
        values = np.array(filtered_df.select("value").rdd.map(lambda row: row[0]).collect())[0]
        return SparseTensorCSR(values, col_indices, crow_indices, original_shape, dense_shape)

    def __read_csc(self, tensor_id: str, slice_tuple: tuple) -> SparseTensorCSC:
        # TODO @evanyfzhou support slicing operation
        df = self.spark.read.format("delta").load(SparkUtil.CSC_TABLE)
        filtered_df = df.filter(df.id == tensor_id)
        dense_shape = filtered_df.select("dense_shape").first()[0]
        ccol_indices = np.array(filtered_df.select(
            "ccol_indices").rdd.map(lambda row: row[0]).collect())[0]
        row_indices = np.array(filtered_df.select(
            "row_indices").rdd.map(lambda row: row[0]).collect())[0]
        values = np.array(filtered_df.select(
            "value").rdd.map(lambda row: row[0]).collect())[0]
        return SparseTensorCSC(values, row_indices, ccol_indices, dense_shape)

    def __read_csf(self, tensor_id: str, slice_tuple: tuple) -> SparseTensorCSF:
        # TODO @kevinvan13 support slicing operation
        # Extract the number of dimensions from the tensor ID
        num_dimensions = int(tensor_id[-2:])  # The first two characters represent the dimensions
        path = f"{SparkUtil.CSF_TABLE}/dim_{num_dimensions}/"
        df = self.spark.read.format("delta").load(path)
        filtered_df = df.filter(df.tensor_id == tensor_id).sort("id")

        # Initialize lists for non-chunked and chunked data
        # Adjust based on column names
        fptrs = [[] for _ in range(max(len(df.columns) // 2 - 3, 2))]
        # Adjust based on column names
        fids = [[] for _ in range(max(len(df.columns) // 2 - 2, 3))]
        values = []

        # Variables for non-chunked data, initially None
        fptr_zero = fptr_one = fid_zero = fid_one = dense_shape = None

        for row in filtered_df.collect():
            row_dict = row.asDict()
            if 'values_chunk' in row_dict and row_dict['values_chunk'] is not None:
                values.extend(row_dict['values_chunk'])


            for i in range(len(fptrs)):
                if f'fptr_{i}_chunk' in row_dict and row_dict[f'fptr_{i}_chunk'] is not None:
                    fptrs[i].extend(row_dict[f'fptr_{i}_chunk'])


            for i in range(len(fids)):
                if f'fid_{i}_chunk' in row_dict and row_dict[f'fid_{i}_chunk'] is not None:
                    fids[i].extend(row_dict[f'fid_{i}_chunk'])

            # Only set non-chunked data if not already set
            if fptr_zero is None:
                fptr_zero = row_dict.get('fptr_zero', [])
                fptr_one = row_dict.get('fptr_one', [])
                fid_zero = row_dict.get('fid_zero', [])
                fid_one = row_dict.get('fid_one', [])
                dense_shape = row_dict.get('dense_shape', [])
        # Ensure dense_shape is correctly formatted if it was ever set
        dense_shape = tuple(dense_shape) if dense_shape is not None else ()

        # Adjust the first two dimensions manually as they are not part of the loop
        fptrs[0], fptrs[1] = fptr_zero or [], fptr_one or []
        fids[0], fids[1] = fid_zero or [], fid_one or []
        values = np.array(values) if values else np.array([])

        return SparseTensorCSF(fptrs=fptrs, fids=fids, values=values, dense_shape=dense_shape, 
                               slice_tuple = self.__parse_slice_tuple(slice_tuple, dense_shape) if slice_tuple else None)


    def __read_mode_generic(self, tensor_id: str, slice_tuple: tuple) -> SparseTensorModeGeneric:
        df = self.spark.read.format("delta").load(SparkUtil.MODE_GENERIC_TABLE)
        filtered_df = df.filter(df.id == tensor_id)
        # filtered_df.show()
        dense_shape, block_shape = filtered_df.select("dense_shape", "block_shape").first()
        slice_tuple = self.__parse_slice_tuple(slice_tuple, dense_shape)

        def filter_predicate(row):
            for i, s in enumerate(slice_tuple):
                if s[0] <= row[0][i] * block_shape[i] < s[1]:
                    continue
                else:
                    return False
            return True

        indices = np.array(
            filtered_df.select("index_array").rdd.filter(filter_predicate).map(
                lambda row: row[0]).collect()).transpose()
        values = [np.array(_) for _ in filtered_df.select("index_array", "value").rdd.filter(filter_predicate).map(
            lambda row: row[1]).collect()]

        return SparseTensorModeGeneric(indices, values, tuple(block_shape), tuple(dense_shape))

    @staticmethod
    def __parse_slice_tuple(slice_tuple: tuple, dense_shape: tuple) -> tuple:
        if len(slice_tuple) > len(dense_shape):
            raise Exception("Invalid slicing operation")
        slice_list = [(0, _) for _ in dense_shape]
        for i in range(len(slice_tuple)):
            if ':' not in slice_tuple[i]:
                slice_list[i] = (int(slice_tuple[i]), int(slice_tuple[i]) + 1)
            else:
                l, _, r = slice_tuple[i].partition(':')
                slice_list[i] = (slice_list[i][0] if not l else max(slice_list[i][0], int(l)),
                                 slice_list[i][1] if not r else min(slice_list[i][1], int(r)))
        return tuple(slice_list)
