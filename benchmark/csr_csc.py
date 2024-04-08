from api.delta_tensor import *
from util.data_util import get_uber_dataset

def example_sparse_tensor2d(delta_tensor: DeltaTensor) -> None:
    indices = np.array([[0, 1, 1, 1],
                        [2, 0, 2, 1]])
    values = np.array([3, 4, 5, -1])
    dense_shape = (2, 4)
    sparse = SparseTensorCOO(indices, values, dense_shape)
    print("original tensor: ", sparse)
    t_id = delta_tensor.save_sparse_tensor(sparse, layout=SparseTensorLayout.CSR)
    print(t_id)
    tensor = delta_tensor.get_sparse_tensor_by_id(t_id, layout=SparseTensorLayout.CSR)
    print("restored tensor: ", tensor)

def example_sparse_tensor_csr_4d(delta_tensor: DeltaTensor) -> None:
    indices = np.array([[0, 0, 1, 1],
                        [0, 1, 0, 1],
                        [1, 1, 1, 1],
                        [0, 2, 2, 3]])
    values = np.array([3, 4, 5, -1])
    dense_shape = (2, 2, 2, 4)
    sparse = SparseTensorCOO(indices, values, dense_shape)
    print("original tensor: ", sparse)
    t_id = delta_tensor.save_sparse_tensor(sparse, layout=SparseTensorLayout.CSR)
    print(t_id)
    tensor = delta_tensor.get_sparse_tensor_by_id(t_id, layout=SparseTensorLayout.CSR)
    print("restored tensor: ", tensor)

def example_sparse_tensor_csc_4d(delta_tensor: DeltaTensor) -> None:
    indices = np.array([[0, 0, 1, 1],
                        [0, 1, 0, 1],
                        [1, 1, 1, 1],
                        [0, 2, 2, 3]])
    values = np.array([3, 4, 5, -1])
    dense_shape = (2, 2, 2, 4)
    sparse = SparseTensorCOO(indices, values, dense_shape)
    print("original tensor: ", sparse)
    t_id = delta_tensor.save_sparse_tensor(sparse, layout=SparseTensorLayout.CSC)
    print(t_id)
    tensor = delta_tensor.get_sparse_tensor_by_id(t_id, layout=SparseTensorLayout.CSC)
    print("restored tensor: ", tensor)

def benchmark_uber_dataset(delta_tensor: DeltaTensor) -> None:
    print("=======================================")
    print("Mode Generic benchmark for uber dataset")
    sparse = get_uber_dataset()
    start = time.time()
    t_id = delta_tensor.save_sparse_tensor(sparse, layout=SparseTensorLayout.CSR)
    print(f"Tensor insertion time: {time.time() - start} seconds")
    start = time.time()
    delta_tensor.get_sparse_tensor_by_id(t_id, layout=SparseTensorLayout.CSR)
    print(f"Tensor retrieving time: {time.time() - start} seconds")

if __name__ == '__main__':
    delta_tensor = DeltaTensor(SparkUtil())

    # Test for sparse tensor
    example_sparse_tensor_csr_4d(delta_tensor)

    # Test for uber set
    benchmark_uber_dataset(delta_tensor)