import numpy as np


class SparseTensorCOO:
    def __init__(self, indices: np.ndarray, values: np.ndarray, dense_shape: tuple):
        self.indices = indices
        self.values = values
        self.dense_shape = dense_shape
        self.layout = "COO"

    def __str__(self):
        return f"SparseTensor(\nindices=\n{self.indices},\nvalues=\n{self.values},\ndense_shape={self.dense_shape},\nlayout={self.layout})"

class SparseTensorCSR:
    def __init__(self, data: np.ndarray, indices: np.ndarray, indptr: np.ndarray, dense_shape: tuple):
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.dense_shape = dense_shape
        self.layout = "CSR"

    def __str__(self):
        return f"SparseTensor(\nlayout={self.layout},\nshape={self.dense_shape},\ndata=\n{self.data},\nindices=\n{self.indices},\nindptr=\n{self.indptr})"

class SparseTensorCSC:
    def __init__(self, data: np.ndarray, indices: np.ndarray, indptr: np.ndarray, dense_shape: tuple):
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.dense_shape = dense_shape
        self.layout = "CSC"

    def __str__(self):
        return f"SparseTensor(\nlayout={self.layout},\nshape={self.dense_shape},\ndata=\n{self.data},\nindices=\n{self.indices},\nindptr=\n{self.indptr})"

class SparseTensorModeGeneric:
    def __init__(self, indices: np.ndarray, values: np.ndarray, block_shape: tuple, dense_shape: tuple):
        self.indices = indices
        self.values = values
        self.block_shape = block_shape
        self.dense_shape = dense_shape
        self.layout = "ModeGeneric"

    def __str__(self):
        return f"SparseTensor(\nindices=\n{self.indices},\nvalues=\n{self.values},\nblock_shape={self.block_shape},\ndense_shape={self.dense_shape},\nlayout={self.layout})"
