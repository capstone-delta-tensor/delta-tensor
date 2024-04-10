import re
import time

from algorithms.sparse import *
from tensor.sparse_tensor import SparseTensorCOO
from util.spark_util import SparkUtil
from typing import overload


class DeltaTensor:
    def __init__(self, spark_util: SparkUtil):
        self.spark_util = spark_util

    def save_dense_tensor(self, tensor: np.ndarray, chunk_dim_count: int = 3) -> str:
        return self.spark_util.write_dense_tensor(tensor, chunk_dim_count)

    def save_dense_tensor_as_sparse(self, tensor: np.ndarray) -> str:
        # TODO support more types
        sparse_tensor = ndarray_to_mode_generic(tensor)
        return self.spark_util.write_sparse_tensor(sparse_tensor)

    def save_sparse_tensor(self, tensor: SparseTensorCOO, layout: SparseTensorLayout, block_shape: tuple = ()) -> str:
        start_time = time.time()
        sparse_tensor = coo_to_sparse(tensor, layout, block_shape)
        print(f"Time to encoding {time.time() - start_time} seconds")
        start_time = time.time()
        id = self.spark_util.write_tensor(sparse_tensor, is_sparse=True)
        print(f"Time to write tensor {time.time() - start_time} seconds")
        return id
    
    @overload
    def get_dense_tensor_by_id(self, id: str, slice: tuple) -> np.ndarray:
        ...

    @overload
    def get_dense_tensor_by_id(self, id: str, slice: str) -> np.ndarray:
        ...

    def get_dense_tensor_by_id(self, id: str, slice = ()) -> np.ndarray:
        if isinstance(slice, str):
            slice = self.__parse_slice_expr(slice)

        self.__slice_tuple_check(slice)
        return self.spark_util.read_dense_tensor(id, slice)

    def get_sparse_tensor_as_dense_by_id(self, id: str) -> np.ndarray:
        # TODO support more types
        sparse_tensor = self.spark_util.read_sparse_tensor(id, layout=SparseTensorLayout.MODE_GENERIC)
        return mode_generic_to_ndarray(sparse_tensor)

    def get_sparse_tensor_by_id(self, id: str, layout: SparseTensorLayout, slice_expr: str = None) -> SparseTensorCOO:
        start_time = time.time()
        sparse_tensor = self.spark_util.read_tensor(id, is_sparse=True, layout=layout,
                                                    slice_tuple=self.__parse_slice_expr(slice_expr))
        print(f"Time to read tensor {time.time() - start_time} seconds")
        start_time = time.time()
        sparse_coo = sparse_to_coo(sparse_tensor)
        print(f"Time to decode tensor {time.time() - start_time} seconds")
        return sparse_coo

    @staticmethod
    def __parse_slice_expr(slice_expr: str) -> tuple:
        return tuple(re.sub(r'[\[\] ]', '', slice_expr).split(',')) if slice_expr else ()

    @staticmethod
    def __slice_tuple_check(slice_tuple: tuple) -> None:
        if not isinstance(slice_tuple, tuple):
            raise ValueError("Input must be a tuple.")
        if slice_tuple == ():
            return
        for i in slice_tuple:
            if not isinstance(i, tuple) or len(i) != 2:
                raise ValueError("Each element of the tuple must be a tuple of length 2.")
            if not isinstance(i[0], int) or not isinstance(i[1], int):
                raise ValueError("Both elements of the inner tuple must be integers.")
            if i[1] <= i[0]:
                raise ValueError("Empty slice")
