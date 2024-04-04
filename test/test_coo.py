import torch
import unittest

from api.delta_tensor import *
import numpy as np
import time
from api.delta_tensor import SparseTensorCOO


class TestCOO(unittest.TestCase):
    def setUp(self):
        self.spark_util = SparkUtil()
        self.delta_tensor = DeltaTensor(self.spark_util)

    def test_example_sparse_tensor(self):
        print("======================================")
        print("COO example for sparse tensor")
        indices = np.array([[0, 1, 1, 1],
                            [2, 0, 2, 1],
                            [0, 1, 2, 1]])
        values = np.array([3, 4, 5, -1])
        dense_shape = (2, 4, 3)
        sparse = SparseTensorCOO(indices, values, dense_shape)
        t_id = self.delta_tensor.save_sparse_tensor(
            sparse, layout=SparseTensorLayout.COO)
        tensor = self.delta_tensor.get_sparse_tensor_by_id(
            t_id, layout=SparseTensorLayout.COO)
        self.assertTrue(sparse == tensor)

    def test_benchmark_uber_dataset(self):
        print("=======================================")
        print("COO for uber dataset")
        uber_sparse_tensor = np.loadtxt(
            "/home/danny/delta-tensor/dataset/uber/uber.tns", dtype=int).transpose()
        indices = uber_sparse_tensor[0:-1]
        values = uber_sparse_tensor[-1]
        dense_shape = (183, 24, 1140, 1717)
        sparse = SparseTensorCOO(indices, values, dense_shape)
        start = time.time()
        t_id = self.delta_tensor.save_sparse_tensor(
            sparse, layout=SparseTensorLayout.COO, block_shape=(4, 4))
        print(f"Tensor insertion time: {time.time() - start} seconds")
        start = time.time()
        retrieved = self.delta_tensor.get_sparse_tensor_by_id(
            t_id, layout=SparseTensorLayout.COO)
        print(f"Tensor retrieving time: {time.time() - start} seconds")
        self.assertTrue(sparse == retrieved)


if __name__ == '__main__':
    unittest.main()