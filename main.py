import numpy as np
import torch

from delta_tensor import *
from spark_util import SparkUtil

if __name__ == '__main__':
    spark_util = SparkUtil()

    # Test for dense tensor
    dense = np.zeros([3, 2, 2, 3, 2])
    dense[0, 0, 0, :, :] = np.arange(6).reshape(3, 2)
    dense[1, 1, 0, :, :] = np.arange(6, 12).reshape(3, 2)
    dense[2, 1, 1, :, :] = np.arange(12, 18).reshape(3, 2)

    t_id = insert_tensor(spark_util, dense)
    print(t_id)
    tensor = find_tensor_by_id(spark_util, t_id)
    print(tensor)
    print(np.array_equal(tensor, dense))

    # Test for sparse tensor
    indices = np.array([[0, 1, 1, 1],
                        [2, 0, 2, 1]])
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

    # Test for uber set
    uber_sparse_tensor = np.loadtxt("dataset/uber/uber.tns", dtype=int).transpose()
    indices = uber_sparse_tensor[0:-1]
    values = uber_sparse_tensor[-1]
    dense_shape = (183, 24, 1140, 1717)
    sparse = SparseTensorCOO(indices, values, dense_shape)
    t_id = insert_sparse_tensor(spark_util, sparse, block_shape=(4, 4))
    print(t_id)
    tensor = find_sparse_tensor_by_id(spark_util, t_id)
    print(tensor)
