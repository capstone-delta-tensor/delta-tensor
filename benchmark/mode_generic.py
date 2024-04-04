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


def example_sparse_tensor_slicing(delta_tensor: DeltaTensor) -> None:
    print("==============================================")
    print("Mode Generic slicing example for sparse tensor")
    indices = np.array([[0, 1, 1, 1],
                        [2, 0, 2, 1],
                        [0, 1, 2, 1]])
    values = np.array([3, 4, 5, -1])
    dense_shape = (2, 4, 3)
    sparse = SparseTensorCOO(indices, values, dense_shape)
    print(sparse)
    t_id = delta_tensor.save_sparse_tensor(sparse, layout=SparseTensorLayout.MODE_GENERIC)
    print(t_id)
    tensor = delta_tensor.get_sparse_tensor_by_id(t_id, layout=SparseTensorLayout.MODE_GENERIC,
                                                  slice_expr='[1, 0:1, :]')
    print(tensor)

    torch_sparse = torch.sparse_coo_tensor(torch.tensor(sparse.indices), torch.tensor(sparse.values), dense_shape)
    torch_result_sparse = torch.sparse_coo_tensor(torch.tensor(tensor.indices), torch.tensor(tensor.values),
                                                  dense_shape)
    print(torch_sparse.to_dense())
    print(torch_result_sparse.to_dense())


def benchmark_uber_dataset_as_pt(sparse: SparseTensorCOO) -> None:
    tensor = torch.sparse_coo_tensor(torch.tensor(sparse.indices), torch.tensor(sparse.values), sparse.dense_shape)
    start = time.time()
    torch.save(tensor, '/tmp/tensor.pt')
    print(f"Tensor saving time: {time.time() - start} seconds")
    start = time.time()
    tensor = torch.load('/tmp/tensor.pt')
    tensor = tensor[0]
    SparseTensorCOO(np.array(tensor.indices), np.array(tensor.values), tensor.size)
    print(f"Tensor loading time: {time.time() - start} seconds")


def benchmark_writing_uber_dataset(delta_tensor: DeltaTensor, sparse: SparseTensorCOO) -> str:
    print("===============================================")
    print("Mode Generic benchmark for writing uber dataset")
    start = time.time()
    t_id = delta_tensor.save_sparse_tensor(sparse, layout=SparseTensorLayout.MODE_GENERIC, block_shape=(4, 4))
    print(f"Tensor insertion time: {time.time() - start} seconds")
    return t_id


def benchmark_reading_uber_dataset(delta_tensor: DeltaTensor, t_id: str) -> None:
    print("===============================================")
    print("Mode Generic benchmark for reading uber dataset")
    start = time.time()
    delta_tensor.get_sparse_tensor_by_id(t_id, layout=SparseTensorLayout.MODE_GENERIC)
    print(f"Tensor full scan time: {time.time() - start} seconds")

    start = time.time()
    delta_tensor.get_sparse_tensor_by_id(t_id, layout=SparseTensorLayout.MODE_GENERIC,
                                         slice_expr='[0, 0:1, 530:540, :]')
    print(f"Tensor slicing time: {time.time() - start} seconds")

    cnt = 183
    start = time.time()
    for i in range(cnt):
        delta_tensor.get_sparse_tensor_by_id(t_id, layout=SparseTensorLayout.MODE_GENERIC,
                                             slice_expr=f'[{i}, 6:18, :, :]')
    time_interval = time.time() - start
    print(
        f"Tensor slicing time for {cnt} iterations: {time_interval} seconds, {time_interval / cnt} seconds per iteration")


if __name__ == '__main__':
    delta_tensor = DeltaTensor(SparkUtil())

    # Test for dense tensor
    example_dense_tensor(delta_tensor)

    # Test for sparse tensor
    example_sparse_tensor(delta_tensor)
    example_sparse_tensor_slicing(delta_tensor)

    # Load uber dataset
    uber_sparse_tensor = np.loadtxt("dataset/uber/uber.tns", dtype=int).transpose()
    indices = uber_sparse_tensor[0:-1] - 1
    values = uber_sparse_tensor[-1]
    dense_shape = (183, 24, 1140, 1717)
    uber_sparse = SparseTensorCOO(indices, values, dense_shape)

    # Test for torch pt file
    benchmark_uber_dataset_as_pt(uber_sparse)

    # # Test for tensor writing
    # t_id = benchmark_writing_uber_dataset(delta_tensor, uber_sparse)
    #
    # # Test for tensor reading
    # benchmark_reading_uber_dataset(delta_tensor, t_id)
