from api.delta_tensor import *
from util.data_util import get_uber_dataset


def example_sparse_tensor(delta_tensor: DeltaTensor) -> None:
    print("======================================")
    print("CSF example for sparse tensor")
    indices = np.array([[0, 1, 1, 1],
                        [2, 0, 2, 1],
                        [0, 1, 2, 1]])
    values = np.array([3, 4, 5, -1])
    dense_shape = (2, 4, 3)
    sparse = SparseTensorCOO(indices, values, dense_shape)
    print(sparse)
    t_id = delta_tensor.save_sparse_tensor(sparse, layout=SparseTensorLayout.CSF)
    print(t_id)
    tensor = delta_tensor.get_sparse_tensor_by_id(t_id, layout=SparseTensorLayout.CSF)
    print(tensor)

    torch_sparse = torch.sparse_coo_tensor(torch.tensor(sparse.indices), torch.tensor(sparse.values), dense_shape)
    torch_result_sparse = torch.sparse_coo_tensor(torch.tensor(tensor.indices), torch.tensor(tensor.values),
                                                  dense_shape)
    print(torch_sparse.to_dense())
    print(torch_result_sparse.to_dense())


def benchmark_uber_dataset(delta_tensor: DeltaTensor) -> None:
    print("=======================================")
    print("CSF benchmark for uber dataset")
    sparse = get_uber_dataset()
    start = time.time()
    t_id = delta_tensor.save_sparse_tensor(sparse, layout=SparseTensorLayout.CSF)
    print(f"Tensor insertion time: {time.time() - start} seconds")
    start = time.time()
    retrieved = delta_tensor.get_sparse_tensor_by_id(t_id, layout=SparseTensorLayout.CSF)
    print(f"Tensor retrieving time: {time.time() - start} seconds")
    print("Data consistency: ", sparse == retrieved)


if __name__ == '__main__':
    delta_tensor = DeltaTensor(SparkUtil())

    # Test for sparse tensor
    example_sparse_tensor(delta_tensor)

    # Test for uber set
    benchmark_uber_dataset(delta_tensor)
