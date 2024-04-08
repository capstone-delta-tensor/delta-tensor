from api.delta_tensor import *
from util.data_util import get_uber_dataset


def benchmark_uber_dataset(delta_tensor: DeltaTensor) -> None:
    print("=======================================")
    print("COO for uber dataset")
    sparse = get_uber_dataset()
    start = time.time()
    t_id = delta_tensor.save_sparse_tensor(
        sparse, layout=SparseTensorLayout.COO, block_shape=(4, 4))
    print(f"Tensor insertion time: {time.time() - start} seconds")
    start = time.time()
    retrieved = delta_tensor.get_sparse_tensor_by_id(
        t_id, layout=SparseTensorLayout.COO)
    print(f"Tensor retrieving time: {time.time() - start} seconds")


if __name__ == '__main__':
    delta_tensor = DeltaTensor(SparkUtil())

    # Test for uber set
    benchmark_uber_dataset(delta_tensor)
