import statistics

from api.delta_tensor import *
from util.data_util import get_uber_dataset


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
    print(f"Data consistency: {np.array_equal(tensor, dense)}")


def example_sparse_tensor(delta_tensor: DeltaTensor) -> None:
    print("======================================")
    print("Mode Generic example for sparse tensor")
    indices = np.array([[0, 1, 1, 1],
                        [2, 0, 1, 2],
                        [0, 1, 1, 2]])
    values = np.array([3, 4, 5, -1]).astype(float)
    dense_shape = (2, 4, 3)
    sparse = SparseTensorCOO(indices, values, dense_shape)
    print(sparse)
    t_id = delta_tensor.save_sparse_tensor(sparse, layout=SparseTensorLayout.MODE_GENERIC)
    print(t_id)
    tensor = delta_tensor.get_sparse_tensor_by_id(t_id, layout=SparseTensorLayout.MODE_GENERIC)
    print(tensor)
    print(f"Data consistency: {tensor == sparse}")


def example_sparse_tensor_slicing(delta_tensor: DeltaTensor) -> None:
    print("==============================================")
    print("Mode Generic slicing example for sparse tensor")
    indices = np.array([[0, 1, 1, 1],
                        [2, 0, 2, 1],
                        [0, 1, 2, 1]])
    values = np.array([3, 4, 5, -1]).astype(float)
    dense_shape = (2, 4, 3)
    sparse = SparseTensorCOO(indices, values, dense_shape)
    print(sparse)
    t_id = delta_tensor.save_sparse_tensor(sparse, layout=SparseTensorLayout.MODE_GENERIC)
    print(t_id)
    tensor = delta_tensor.get_sparse_tensor_by_id(t_id, layout=SparseTensorLayout.MODE_GENERIC,
                                                  slice_expr='[1, 0:1, :]')
    print(tensor)


def benchmark_writing_uber_dataset(delta_tensor: DeltaTensor, sparse: SparseTensorCOO) -> tuple[str, float]:
    print("===============================================")
    print("Mode Generic benchmark for writing uber dataset")
    start = time.time()
    t_id = delta_tensor.save_sparse_tensor(sparse, layout=SparseTensorLayout.MODE_GENERIC, block_shape=(12, 256, 256))
    insertion_time = time.time() - start
    print(f"Tensor insertion time: {insertion_time} seconds")
    return t_id, insertion_time


def benchmark_reading_uber_dataset(delta_tensor: DeltaTensor, t_id: str) -> tuple[float, float]:
    print("===============================================")
    print("Mode Generic benchmark for reading uber dataset")
    start = time.time()
    sparse = delta_tensor.get_sparse_tensor_by_id(t_id, layout=SparseTensorLayout.MODE_GENERIC)
    full_scan_time = time.time() - start
    print(f"Tensor full scan time: {full_scan_time} seconds")
    print(f"Data consistency: {sparse == uber_sparse}")

    cnt = 10
    start = time.time()
    for i in range(cnt):
        delta_tensor.get_sparse_tensor_by_id(t_id, layout=SparseTensorLayout.MODE_GENERIC,
                                             slice_expr=f'[{i}, :, :, :]')
    time_interval = time.time() - start
    avg_slicing_time = time_interval / cnt
    print(
        f"Tensor slicing time for {cnt} iterations: {time_interval} seconds, {avg_slicing_time} seconds per iteration")

    return full_scan_time, avg_slicing_time


if __name__ == '__main__':
    delta_tensor = DeltaTensor(SparkUtil())

    # Test for dense tensor
    example_dense_tensor(delta_tensor)

    # Test for sparse tensor
    example_sparse_tensor(delta_tensor)
    example_sparse_tensor_slicing(delta_tensor)

    # Load uber dataset
    uber_sparse = get_uber_dataset()

    # Epoch number
    epoch = 10

    # Test for tensor writing
    writing_stats = [benchmark_writing_uber_dataset(delta_tensor, uber_sparse) for _ in range(epoch)]

    # Test for tensor reading
    reading_stats = [benchmark_reading_uber_dataset(delta_tensor, _[0]) for _ in writing_stats]

    # Display statistics
    print(f"=====Uber dataset benchmark results for MODE_GENERIC=====")
    print(f"Average tensor insertion time for {epoch} epochs: {statistics.mean([_[1] for _ in writing_stats])}")
    print(f"Average tensor full scan time for {epoch} epochs: {statistics.mean([_[0] for _ in reading_stats])}")
    print(f"Average tensor slicing time for {epoch} epochs: {statistics.mean([_[1] for _ in reading_stats])}")
