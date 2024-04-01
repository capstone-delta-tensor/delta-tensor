import math
import torch
from collections import defaultdict

from tensor.sparse_tensor import *

MAX_BLOCK_SIZE = 1024 * 32


def ndarray_to_mode_generic(tensor: np.ndarray) -> SparseTensorModeGeneric:
    indices_shape, block_shape = __get_block_shapes(tensor.shape)
    indices_size = __get_size_from_shape(indices_shape)
    block_size = __get_size_from_shape(block_shape)

    reshaped_tensor = tensor.reshape(indices_size, block_size)
    sparse_filter = np.apply_along_axis(lambda blk: blk.any(), 1, reshaped_tensor)
    indices = np.apply_along_axis(lambda row: row[sparse_filter], 1,
                                  np.indices(indices_shape).reshape(len(indices_shape), -1))
    values = reshaped_tensor[sparse_filter]
    return SparseTensorModeGeneric(indices, values, block_shape, tensor.shape)


def coo_to_sparse(tensor: SparseTensorCOO, layout: SparseTensorLayout = SparseTensorLayout.MODE_GENERIC,
                  block_shape: tuple = ()) -> SparseTensorCOO | SparseTensorCSR | SparseTensorCSC | SparseTensorCSF | SparseTensorModeGeneric:
    match layout:
        case SparseTensorLayout.COO:
            return tensor
        case SparseTensorLayout.CSR:
            return coo_to_csr(tensor)
        case SparseTensorLayout.CSC:
            return coo_to_csc(tensor)
        case SparseTensorLayout.CSF:
            return coo_to_csf(tensor)
        case SparseTensorLayout.MODE_GENERIC:
            return coo_to_mode_generic(tensor, block_shape)
        case _:
            raise ValueError(f"Layout {layout} not supported")


def coo_to_csr(tensor: SparseTensorCOO) -> SparseTensorCSR:
    coo = torch.sparse_coo_tensor(tensor.indices, tensor.values, tensor.dense_shape, dtype=torch.float32)
    csr = coo.to_sparse_csr()
    return SparseTensorCSR(csr.values().numpy(), csr.col_indices().numpy(), csr.crow_indices().numpy(), tensor.dense_shape)

def coo_to_csc(tensor: SparseTensorCOO) -> SparseTensorCSC:
    coo = torch.sparse_coo_tensor(tensor.indices, tensor.values, tensor.dense_shape, dtype=torch.float32)
    csc = coo.to_sparse_csc()
    return SparseTensorCSC(csc.values().numpy(), csc.row_indices.numpy(), csc.ccol_indices().numpy(), tensor.dense_shape)


def coo_to_csf(tensor: SparseTensorCOO) -> SparseTensorCSF:
    # TODO @kevinvan13
    raise Exception("Not implemented")


def coo_to_mode_generic(tensor: SparseTensorCOO, block_shape: tuple = ()) -> SparseTensorModeGeneric:
    if len(block_shape) == 0:
        _, block_shape = __get_block_shapes(tensor.dense_shape, is_sparse=True)
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

    indices = np.zeros((len(indices_dict), len(block_shape)), dtype=int)
    values = np.zeros((len(indices_dict), __get_size_from_shape(block_shape)))
    for i, key in enumerate(indices_dict):
        indices[i] = indices_dict[key]
        values[i] = blocks_dict[key].reshape(-1)
    return SparseTensorModeGeneric(indices.transpose(), values, block_shape,
                                   tensor.dense_shape)


def mode_generic_to_ndarray(sparse_tensor: SparseTensorModeGeneric) -> np.ndarray:
    indices = sparse_tensor.indices
    values = sparse_tensor.values
    block_shape = sparse_tensor.block_shape
    block_size = __get_size_from_shape(block_shape)
    dense_shape = sparse_tensor.dense_shape

    if len(block_shape) == 0:
        indices_shape = dense_shape
    elif len(block_shape) == len(dense_shape):
        indices_shape = (1,)
    else:
        indices_shape = dense_shape[:-len(block_shape)]
    indices_size = __get_size_from_shape(indices_shape)
    mul = [1] * len(indices_shape)
    for i in range(len(mul) - 1, 0, -1):
        mul[i - 1] = mul[i] * indices_shape[i]
    sparse_filter = np.sum(indices * np.array(mul).reshape(-1, 1), axis=0)

    tensor = np.zeros((indices_size, block_size))
    tensor[sparse_filter] = values
    return tensor.reshape(dense_shape)


def sparse_to_coo(
        sparse_tensor: SparseTensorCOO | SparseTensorCSR | SparseTensorCSC | SparseTensorCSF | SparseTensorModeGeneric) -> SparseTensorCOO:
    match sparse_tensor.layout:
        case SparseTensorLayout.COO:
            return sparse_tensor
        case SparseTensorLayout.CSR:
            return csr_to_coo(sparse_tensor)
        case SparseTensorLayout.CSC:
            return csc_to_coo(sparse_tensor)
        case SparseTensorLayout.CSF:
            return csf_to_coo(sparse_tensor)
        case SparseTensorLayout.MODE_GENERIC:
            return mode_generic_to_coo(sparse_tensor)
        case _:
            raise ValueError(f"Layout {sparse_tensor.layout} not supported")


def csr_to_coo(sparse_tensor: SparseTensorCSR) -> SparseTensorCOO:
    csr = torch.sparse_csr_tensor(sparse_tensor.crow_indices, sparse_tensor.col_indices, sparse_tensor.values, sparse_tensor.dense_shape, dtype=torch.float32)
    coo = csr.to_sparse_coo()
    indices = coo.indices().numpy()
    values = coo.values().numpy()    
    return SparseTensorCOO(indices, values, sparse_tensor.dense_shape)


def csc_to_coo(sparse_tensor: SparseTensorCSC) -> SparseTensorCOO:
    csc = torch.sparse_csc_tensor(sparse_tensor.ccol_indices, sparse_tensor.row_indices, sparse_tensor.values, sparse_tensor.dense_shape, dtype=torch.float32)
    coo = csc.to_sparse_coo()
    indices = coo.indices().numpy()
    values = coo.values().numpy()
    return SparseTensorCOO(indices, values, sparse_tensor.dense_shape)

def csf_to_coo(sparse_tensor: SparseTensorCSF) -> SparseTensorCOO:
    # TODO @kevinvan13
    raise Exception("Not implemented")


def mode_generic_to_coo(sparse_tensor: SparseTensorModeGeneric) -> SparseTensorCOO:
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


def __get_block_shapes(tensor_shape: tuple, is_sparse: bool = False) -> tuple:
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


def __get_size_from_shape(shape: tuple) -> int:
    return np.array(shape).prod() if len(shape) != 0 else 1