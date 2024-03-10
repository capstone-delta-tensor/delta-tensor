import uuid

import numpy as np

from sparse_tensor import SparseTensorBSR

sparse_tensors = {}

page_size = 12


def insert_tensor(tensor: np.ndarray):
    indices_shape, block_shape = get_block_shapes(tensor.shape)
    indices_size = get_size_from_shape(indices_shape)
    block_size = get_size_from_shape(block_shape)

    reshaped_tensor = tensor.reshape(indices_size, block_size)
    sparse_filter = np.apply_along_axis(lambda blk: blk.any(), 1, reshaped_tensor)
    # print(sparse_filter)
    indices = np.apply_along_axis(lambda row: row[sparse_filter], 1,
                                  np.indices(indices_shape).reshape(len(indices_shape), -1))
    values = reshaped_tensor[sparse_filter]
    sparse_tensor = SparseTensorBSR(indices, values, block_shape, tensor.shape)
    print(sparse_tensor)

    id = uuid.uuid4()
    sparse_tensors[id] = sparse_tensor
    return id


def find_tensor_by_id(id: uuid):
    sparse_tensor = sparse_tensors[id]
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
    while l < r:
        mid = l + (r - l) // 2
        if block_sizes[mid] > page_size:
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


if __name__ == '__main__':
    dense = np.zeros([3, 2, 2, 3, 2])
    dense[0, 0, 0, :, :] = np.arange(6).reshape(3, 2)
    dense[1, 1, 0, :, :] = np.arange(6, 12).reshape(3, 2)
    dense[2, 1, 1, :, :] = np.arange(12, 18).reshape(3, 2)
    t_id = insert_tensor(dense)
    tensor = find_tensor_by_id(t_id)
    print(np.array_equal(tensor, dense))
    # print(insert_tensor(dense))
    # dense[0:2, 0] = dense[0:2, 2] = dense[2:4, 1] = 1
    # dense.to_sparse_bsr((2, 1))
