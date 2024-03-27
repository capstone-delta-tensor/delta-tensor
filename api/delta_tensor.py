from algorithms.mode_generic import *
from spark_util.spark_util_mode_generic import SparkUtil
from tensor.sparse_tensor import SparseTensorCOO


def insert_tensor(spark_util: SparkUtil, tensor: np.ndarray) -> str:
    sparse_tensor = create_mode_generic_from_ndarray(tensor)
    return spark_util.write_tensor(sparse_tensor)


def insert_sparse_tensor(spark_util: SparkUtil, tensor: SparseTensorCOO, block_shape: tuple = ()) -> str:
    sparse_tensor = create_mode_generic_from_coo(tensor, block_shape)
    return spark_util.write_tensor(sparse_tensor)


def find_tensor_by_id(spark_util: SparkUtil, id: str) -> np.ndarray:
    sparse_tensor = spark_util.read_tensor(id)
    return create_ndarray_from_mode_generic(sparse_tensor)


def find_sparse_tensor_by_id(spark_util: SparkUtil, id: str) -> SparseTensorCOO:
    sparse_tensor = spark_util.read_tensor(id)
    return create_coo_from_mode_generic(sparse_tensor)
