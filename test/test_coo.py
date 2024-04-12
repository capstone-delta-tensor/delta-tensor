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

    # def test_example_sparse_tensor(self):
    #     print("======================================")
    #     print("COO example for sparse tensor")
    #     indices = np.array([[0, 1, 1, 1],
    #                         [2, 0, 2, 1],
    #                         [0, 1, 2, 1]])
    #     values = np.array([3, 4, 5, -1])
    #     dense_shape = (2, 4, 3)
    #     sparse = SparseTensorCOO(indices, values, dense_shape)
    #     t_id = self.delta_tensor.save_sparse_tensor(
    #         sparse, layout=SparseTensorLayout.COO)
    #     tensor = self.delta_tensor.get_sparse_tensor_by_id(
    #         t_id, layout=SparseTensorLayout.COO)
    #     order = np.ravel_multi_index(sparse.indices, sparse.dense_shape).argsort()
    #     sparse.indices = sparse.indices[:, order]
    #     sparse.values = sparse.values[order]
    #     self.assertTrue(sparse == tensor)

    def test_example_sparse_tensor_slicing(self) -> None:
        print("==============================================")
        print("COO example for sparse tensor slicing")
        indices = np.array([[0, 1, 1, 1],
                            [2, 0, 2, 1],
                            [0, 1, 2, 1]])
        values = np.array([3, 4, 5, -1]).astype(float)
        dense_shape = (2, 4, 3)
        sparse = SparseTensorCOO(indices, values, dense_shape)
        t_id = self.delta_tensor.save_sparse_tensor(
            sparse, layout=SparseTensorLayout.COO)
        tensor = self.delta_tensor.get_sparse_tensor_by_id(t_id, layout=SparseTensorLayout.COO,
                                                           slice_expr='[1, 0:1, :]')
        order = np.ravel_multi_index(
            sparse.indices, sparse.dense_shape).argsort()
        sparse.indices = sparse.indices[:, order]
        sparse.values = sparse.values[order]
        print(tensor)


if __name__ == '__main__':
    unittest.main()
