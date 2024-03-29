import torch

from api.delta_tensor import *


def example_dense_tensor(delta_tensor: DeltaTensor) -> None:
    print("=====================================")
    print("Mode Generic example for dense tensor")
    dense = np.zeros([3, 2, 2, 3, 2])
    dense[0, 0, 0, :, :] = np.arange(6).reshape(3, 2)
    dense[1, 1, 0, :, :] = np.arange(6, 12).reshape(3, 2)
    dense[2, 1, 1, :, :] = np.arange(12, 18).reshape(3, 2)

    t_id = delta_tensor.save_dense_tensor_as_sparse(dense)
    print(t_id)
    tensor = delta_tensor.get_sparse_tensor_as_dense_by_id(t_id)
    print(tensor)
    print(np.array_equal(tensor, dense))


def example_sparse_tensor(delta_tensor: DeltaTensor) -> None:
    print("======================================")
    print("Mode Generic example for sparse tensor")
    indices = np.array([[0, 1, 1, 1],
                        [2, 0, 2, 1],
                        [0, 1, 2, 1]])
    values = np.array([3, 4, 5, -1])
    dense_shape = (2, 4, 3)
    sparse = SparseTensorCOO(indices, values, dense_shape)
    print(sparse)
    t_id = delta_tensor.save_sparse_tensor(sparse, layout=SparseTensorLayout.MODE_GENERIC)
    print(t_id)
    tensor = delta_tensor.get_sparse_tensor_by_id(t_id, layout=SparseTensorLayout.MODE_GENERIC)
    print(tensor)

    torch_sparse = torch.sparse_coo_tensor(torch.tensor(sparse.indices), torch.tensor(sparse.values), dense_shape)
    torch_result_sparse = torch.sparse_coo_tensor(torch.tensor(tensor.indices), torch.tensor(tensor.values),
                                                  dense_shape)
    print(torch_sparse.to_dense())
    print(torch_result_sparse.to_dense())


def benchmark_uber_dataset(delta_tensor: DeltaTensor) -> None:
    print("=======================================")
    print("Mode Generic benchmark for uber dataset")
    uber_sparse_tensor = np.loadtxt("dataset/uber/uber.tns", dtype=int).transpose()
    indices = uber_sparse_tensor[0:-1]
    values = uber_sparse_tensor[-1]
    dense_shape = (183, 24, 1140, 1717)
    sparse = SparseTensorCOO(indices, values, dense_shape)
    start = time.time()
    t_id = delta_tensor.save_sparse_tensor(sparse, layout=SparseTensorLayout.MODE_GENERIC, block_shape=(4, 4))
    print(f"Tensor insertion time: {time.time() - start} seconds")
    start = time.time()
    delta_tensor.get_sparse_tensor_by_id(t_id, layout=SparseTensorLayout.MODE_GENERIC)
    print(f"Tensor retrieving time: {time.time() - start} seconds")


if __name__ == '__main__':
    delta_tensor = DeltaTensor(SparkUtil())

    # Test for dense tensor
    example_dense_tensor(delta_tensor)

    # Test for sparse tensor
    example_sparse_tensor(delta_tensor)

    # Test for uber set
    benchmark_uber_dataset(delta_tensor)
