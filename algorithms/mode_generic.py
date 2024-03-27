import math
from collections import defaultdict

from tensor.sparse_tensor import *

MAX_BLOCK_SIZE = 1024 * 32


def create_mode_generic_from_ndarray(tensor: np.ndarray) -> SparseTensorModeGeneric:
    indices_shape, block_shape = get_block_shapes(tensor.shape)
    indices_size = get_size_from_shape(indices_shape)
    block_size = get_size_from_shape(block_shape)

    reshaped_tensor = tensor.reshape(indices_size, block_size)
    sparse_filter = np.apply_along_axis(lambda blk: blk.any(), 1, reshaped_tensor)
    indices = np.apply_along_axis(lambda row: row[sparse_filter], 1,
                                  np.indices(indices_shape).reshape(len(indices_shape), -1))
    values = reshaped_tensor[sparse_filter]
    return SparseTensorModeGeneric(indices, values, block_shape, tensor.shape)


def create_mode_generic_from_coo(tensor: SparseTensorCOO, block_shape: tuple = ()) -> SparseTensorModeGeneric:
    if len(block_shape) == 0:
        _, block_shape = get_block_shapes(tensor.dense_shape, is_sparse=True)
    elif len(block_shape) != len(tensor.dense_shape):
        diff = len(tensor.dense_shape) - len(block_shape)
        block_shape = [1 if i < diff else block_shape[i - diff] for i in range(len(tensor.dense_shape))]
    elif len(block_shape) > len(tensor.dense_shape):
        raise Exception("Invalid block shape")

    indices_dict = {}
    blocks_dict = defaultdict(lambda: np.zeros(block_shape))
    for i in range(tensor.indices.shape[1]):
        block_indices = tensor.indices[:, i] // block_shape
        value_indices = tensor.indices[:, i] % block_shape
        key = block_indices.tobytes()
        if key not in indices_dict:
            indices_dict[key] = block_indices
        blocks_dict[key][tuple(value_indices)] = tensor.values[i]

    indices = []
    values = []
    for key in indices_dict:
        indices.append(indices_dict[key])
        values.append(blocks_dict[key].reshape(-1))
    return SparseTensorModeGeneric(np.array(indices).transpose(), np.array(values), block_shape,
                                   tensor.dense_shape)


def create_ndarray_from_mode_generic(sparse_tensor: SparseTensorModeGeneric) -> np.ndarray:
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


def create_coo_from_mode_generic(sparse_tensor: SparseTensorModeGeneric) -> SparseTensorCOO:
    indices = sparse_tensor.indices
    values = sparse_tensor.values
    block_shape = sparse_tensor.block_shape
    dense_shape = sparse_tensor.dense_shape

    indices_coo = []
    values_coo = []
    for i in range(indices.shape[1]):
        global_base_indices = indices[:, i] * block_shape
        block = values[i]
        for j, val in enumerate(block):
            if val == 0: continue
            indices_coo.append(global_base_indices + np.unravel_index(j, block_shape))
            values_coo.append(val)

    return SparseTensorCOO(np.array(indices_coo).transpose(), np.array(values_coo), dense_shape)


def get_block_shapes(tensor_shape: tuple, is_sparse: bool = False) -> tuple:
    if is_sparse:
        indices_shape = []
        block_shape = []
        for i in tensor_shape:
            blk = math.floor(math.sqrt(i))
            block_shape.append(blk)
            indices_shape.append(math.ceil(i / blk))
        return indices_shape, block_shape
    else:
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
