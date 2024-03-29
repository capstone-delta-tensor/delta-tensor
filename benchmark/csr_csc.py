from api.delta_tensor import *
from util.spark_util import SparkUtil


def example_sparse_tensor(delta_tensor: DeltaTensor) -> None:
    indices = np.array([[0, 1, 1, 1],
                        [2, 0, 2, 1]])
    values = np.array([3, 4, 5, -1])
    dense_shape = (2, 4)
    sparse = SparseTensorCOO(indices, values, dense_shape)
    print(sparse)
    t_id = delta_tensor.save_sparse_tensor(sparse, layout=SparseTensorLayout.CSR)
    print(t_id)
    tensor = delta_tensor.get_sparse_tensor_by_id(t_id, layout=SparseTensorLayout.CSR)
    print(tensor)


def benchmark_uber_dataset(delta_tensor: DeltaTensor) -> None:
    pass


if __name__ == '__main__':
    delta_tensor = DeltaTensor(SparkUtil())

    # Test for sparse tensor
    example_sparse_tensor(delta_tensor)

    # Test for uber set
    benchmark_uber_dataset(delta_tensor)
