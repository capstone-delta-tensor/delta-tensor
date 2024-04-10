import statistics

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


def example_sparse_tensor_slicing(delta_tensor: DeltaTensor) -> None:
    print("==============================================")
    print("CSF slicing example for sparse tensor")
    indices = np.array([[0, 1, 1, 1],
                        [2, 0, 2, 1],
                        [0, 1, 2, 1]])
    values = np.array([3, 4, 5, -1])
    dense_shape = (2, 4, 3)
    sparse = SparseTensorCOO(indices, values, dense_shape)
    print(sparse)
    t_id = delta_tensor.save_sparse_tensor(sparse, layout=SparseTensorLayout.CSF)
    print(t_id)
    tensor = delta_tensor.get_sparse_tensor_by_id(t_id, layout=SparseTensorLayout.CSF,
                                                  slice_expr='[1, :, :]')
    print(tensor)


def benchmark_uber_dataset(delta_tensor: DeltaTensor) -> tuple[float, float, float]:
    print("=======================================")
    print("CSF benchmark for uber dataset")
    sparse = get_uber_dataset()
    start = time.time()
    t_id = delta_tensor.save_sparse_tensor(sparse, layout=SparseTensorLayout.CSF)
    insertion_time = time.time() - start
    print(f"Tensor insertion time: {insertion_time} seconds")
    start = time.time()
    retrieved = delta_tensor.get_sparse_tensor_by_id(t_id, layout=SparseTensorLayout.CSF)
    reading_time = time.time() - start
    print(f"Tensor retrieving time: {reading_time} seconds")
    print(f"Data consistency: {sparse == retrieved}")

    cnt = 10
    start = time.time()
    for i in range(cnt):
        delta_tensor.get_sparse_tensor_by_id(t_id, layout=SparseTensorLayout.CSF,
                                             slice_expr=f'[{i}, :, :, :]')
    time_interval = time.time() - start
    avg_slicing_time = time_interval / cnt
    print(
        f"Tensor slicing time for {cnt} iterations: {time_interval} seconds, {avg_slicing_time} seconds per iteration")
    return insertion_time, reading_time, avg_slicing_time


if __name__ == '__main__':
    delta_tensor = DeltaTensor(SparkUtil())

    # Test for sparse tensor
    example_sparse_tensor(delta_tensor)

    # Test for slicing
    example_sparse_tensor_slicing(delta_tensor)

    # Epoch number
    epoch = 10

    # Test for uber set
    stats = [benchmark_uber_dataset(delta_tensor) for _ in range(epoch)]

    # Display statistics
    print(f"=====Uber dataset benchmark results for CSF=====")
    print(f"Average tensor insertion time for {epoch} epochs: {statistics.mean([_[0] for _ in stats])}")
    print(f"Average tensor full scan time for {epoch} epochs: {statistics.mean([_[1] for _ in stats])}")
    print(f"Average tensor slicing time for {epoch} epochs: {statistics.mean([_[2] for _ in stats])}")
