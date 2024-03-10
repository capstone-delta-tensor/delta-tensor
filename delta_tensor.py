import numpy as np

from spark_util import SparkUtil
from sparse_tensor_mode_generic import SparseTensorModeGeneric

sparse_tensors = {}

MAX_BLOCK_SIZE = 1024


def insert_tensor(spark_util: SparkUtil, tensor: np.ndarray) -> str:
    indices_shape, block_shape = get_block_shapes(tensor.shape)
    indices_size = get_size_from_shape(indices_shape)
    block_size = get_size_from_shape(block_shape)

    reshaped_tensor = tensor.reshape(indices_size, block_size)
    sparse_filter = np.apply_along_axis(lambda blk: blk.any(), 1, reshaped_tensor)
    indices = np.apply_along_axis(lambda row: row[sparse_filter], 1,
                                  np.indices(indices_shape).reshape(len(indices_shape), -1))
    values = reshaped_tensor[sparse_filter]
    sparse_tensor = SparseTensorModeGeneric(indices, values, block_shape, tensor.shape)
    print(sparse_tensor)

    return spark_util.write_tensor(sparse_tensor)


def find_tensor_by_id(spark_util: SparkUtil, id: str) -> np.ndarray:
    sparse_tensor = spark_util.read_tensor(id)
    indices = sparse_tensor.indices
    values = sparse_tensor.values
    block_shape = sparse_tensor.block_shape
    block_size = get_size_from_shape(block_shape)
    dense_shape = sparse_tensor.dense_shape

    if len(block_shape) == 0:
        indices_shape = dense_shape
    elif len(block_shape) == len(dense_shape):
        indices_shape = (1,)
    else:
        indices_shape = dense_shape[:-len(block_shape)]
    indices_size = get_size_from_shape(indices_shape)
    mul = [1] * len(indices_shape)
    for i in range(len(mul) - 1, 0, -1):
        mul[i - 1] = mul[i] * indices_shape[i]
    sparse_filter = np.sum(indices * np.array(mul).reshape(-1, 1), axis=0)

    tensor = np.zeros((indices_size, block_size))
    tensor[sparse_filter] = values
    return tensor.reshape(dense_shape)


def get_block_shapes(tensor_shape: tuple) -> tuple:
    block_sizes = [1] * (len(tensor_shape) + 1)
    for i in range(len(tensor_shape) - 1, -1, -1):
        block_sizes[i] = block_sizes[i + 1] * tensor_shape[i]

    l, r = 0, len(tensor_shape)
    mid = l + (r - l) // 2
    if block_sizes[mid] < MAX_BLOCK_SIZE:
        pivot = mid
    else:
        while l < r:
            mid = l + (r - l) // 2
            if block_sizes[mid] > MAX_BLOCK_SIZE:
                l = mid + 1
            else:
                r = mid
        pivot = l

    if pivot <= 0:
        return (1,), tensor_shape
    elif pivot >= len(tensor_shape):
        return tensor_shape, ()
    return tensor_shape[:pivot], tensor_shape[pivot:]


def get_size_from_shape(shape: tuple) -> int:
    return np.array(shape).prod() if len(shape) != 0 else 1
