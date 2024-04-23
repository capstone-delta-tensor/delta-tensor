import statistics

from api.delta_tensor import *
from util.data_util import get_uber_dataset


def benchmark_uber_dataset(delta_tensor: DeltaTensor) -> tuple[float, float, float]:
    print("=======================================")
    print("COO benchmark for uber dataset")
    sparse = get_uber_dataset()
    start = time.time()
    t_id = delta_tensor.save_sparse_tensor(sparse, layout=SparseTensorLayout.COO)
    insertion_time = time.time() - start
    print(f"Tensor insertion time: {insertion_time} seconds")
    start = time.time()
    delta_tensor.get_sparse_tensor_by_id(t_id, layout=SparseTensorLayout.COO)
    reading_time = time.time() - start
    print(f"Tensor retrieving time: {reading_time} seconds")
    cnt = 10
    start = time.time()
    for i in range(cnt):
        delta_tensor.get_sparse_tensor_by_id(t_id, layout=SparseTensorLayout.COO,
                                             slice_expr=f'[{i}, :, :, :]')
    time_interval = time.time() - start
    avg_slicing_time = time_interval / cnt
    print(
        f"Tensor slicing time for {cnt} iterations: {time_interval} seconds, {avg_slicing_time} seconds per iteration")
    return insertion_time, reading_time, avg_slicing_time


if __name__ == '__main__':
    delta_tensor = DeltaTensor(SparkUtil())

    # Epoch number
    epoch = 10

    # Test for uber set
    stats = [benchmark_uber_dataset(delta_tensor) for _ in range(epoch)]

    # Display statistics
    print(f"=====Uber dataset benchmark results for COO=====")
    print(f"Average tensor insertion time for {epoch} epochs: {statistics.mean([_[0] for _ in stats])}")
    print(f"Average tensor full scan time for {epoch} epochs: {statistics.mean([_[1] for _ in stats])}")
    print(f"Average tensor slicing time for {epoch} epochs: {statistics.mean([_[2] for _ in stats])}")
