import time

import torch

from api.delta_tensor import *


def example_dense_tensor(spark_util: SparkUtil):
    dense = np.zeros([3, 2, 2, 3, 2])
    dense[0, 0, 0, :, :] = np.arange(6).reshape(3, 2)
    dense[1, 1, 0, :, :] = np.arange(6, 12).reshape(3, 2)
    dense[2, 1, 1, :, :] = np.arange(12, 18).reshape(3, 2)

    t_id = insert_tensor(spark_util, dense)
    print(t_id)
    tensor = find_tensor_by_id(spark_util, t_id)
    print(tensor)
    print(np.array_equal(tensor, dense))


def example_sparse_tensor(spark_util: SparkUtil):
    indices = np.array([[0, 1, 1, 1],
                        [2, 0, 2, 1],
                        [0, 1, 2, 0]])
    values = np.array([3, 4, 5, -1])
    dense_shape = (2, 4, 3)
    sparse = SparseTensorCOO(indices, values, dense_shape)
    print(sparse)
    t_id = insert_sparse_tensor(spark_util, sparse)
    print(t_id)
    tensor = find_sparse_tensor_by_id(spark_util, t_id)
    print(tensor)

    torch_sparse = torch.sparse_coo_tensor(torch.tensor(sparse.indices), torch.tensor(sparse.values), dense_shape)
    torch_result_sparse = torch.sparse_coo_tensor(torch.tensor(tensor.indices), torch.tensor(tensor.values),
                                                  dense_shape)
    print(torch_sparse.to_dense())
    print(torch_result_sparse.to_dense())

def example_csr_tensor(spark_util: SparkUtil):
    indices = np.array([[0, 0, 3], [4, 0, 0], [0, 6, 0]])
    sparse_csr = create_csr_from_ndarray(indices)
    print(sparse_csr)
    t_id = insert_sparse_tensor(spark_util, sparse_csr)
    print(t_id)
    tensor = find_sparse_tensor_by_id(spark_util, t_id)
    print(tensor)

def example_csc_tensor(spark_util: SparkUtil):
    indices = np.array([[0, 0, 3], [4, 0, 0], [0, 6, 0]])
    sparse_csc = create_csc_from_ndarray(indices)
    print(sparse_csc)
    t_id = insert_sparse_tensor(spark_util, sparse_csc)
    print(t_id)
    tensor = find_sparse_tensor_by_id(spark_util, t_id)
    print(tensor)

def benchmark_uber_dataset(spark_util: SparkUtil):
    uber_sparse_tensor = np.loadtxt("/Users/evan/Downloads/dataset/uber/uber.tns", dtype=int).transpose()
    indices = uber_sparse_tensor[0:-1]
    values = uber_sparse_tensor[-1]
    dense_shape = (183, 24, 1140, 1717)
    sparse = SparseTensorCOO(indices, values, dense_shape)
    start = time.time()
    t_id = insert_sparse_tensor(spark_util, sparse, block_shape=(2, 2))
    print(f"Tensor insertion time: {time.time() - start}")
    print(t_id)
    start = time.time()
    tensor = find_sparse_tensor_by_id(spark_util, t_id)
    print(f"Tensor retrieving time: {time.time() - start}")
    print(tensor)
    print(len(tensor.values))


if __name__ == '__main__':
    spark_util = SparkUtil()

    # Test for dense tensor
    #example_dense_tensor(spark_util)

    # Test for sparse tensor
    #example_sparse_tensor(spark_util)

    # Test for uber set
    #benchmark_uber_dataset(spark_util)
    # Test for csr and csc
    example_csr_tensor(spark_util)
    example_csc_tensor(spark_util)