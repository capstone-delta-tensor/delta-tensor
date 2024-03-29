from enum import Enum

import numpy as np


class SparseTensorLayout(Enum):
    COO = 1
    CSR = 2
    CSC = 3
    CSF = 4
    MODE_GENERIC = 5


class SparseTensorCOO:
    def __init__(self, indices: np.ndarray, values: np.ndarray, dense_shape: tuple):
        self.indices = indices
        self.values = values
        self.dense_shape = dense_shape
        self.layout = SparseTensorLayout.COO

    def __str__(self):
        return f"SparseTensor(\nindices=\n{self.indices},\nvalues=\n{self.values},\ndense_shape={self.dense_shape},\nlayout={self.layout})"


class SparseTensorCSR:
    def __init__(self, dense_shape: tuple):
        # TODO @evanyfzhou Add fields as needed
        self.dense_shape = dense_shape
        self.layout = SparseTensorLayout.CSR

    def __str__(self):
        return f"SparseTensor(\ndense_shape={self.dense_shape},\nlayout={self.layout})"


class SparseTensorCSC:
    def __init__(self, dense_shape: tuple):
        # TODO @evanyfzhou Add fields as needed
        self.dense_shape = dense_shape
        self.layout = SparseTensorLayout.CSC

    def __str__(self):
        return f"SparseTensor(\ndense_shape={self.dense_shape},\nlayout={self.layout})"


class SparseTensorCSF:
    def __init__(self, dense_shape: tuple):
        # TODO @kevinvan13 Add fields as needed
        self.dense_shape = dense_shape
        self.layout = SparseTensorLayout.CSF

    def __str__(self):
        return f"SparseTensor(\ndense_shape={self.dense_shape},\nlayout={self.layout})"


class SparseTensorModeGeneric:
    def __init__(self, indices: np.ndarray, values: np.ndarray, block_shape: tuple, dense_shape: tuple):
        self.indices = indices
        self.values = values
        self.block_shape = block_shape
        self.dense_shape = dense_shape
        self.layout = SparseTensorLayout.MODE_GENERIC

    def __str__(self):
        return f"SparseTensor(\nindices=\n{self.indices},\nvalues=\n{self.values},\nblock_shape={self.block_shape},\ndense_shape={self.dense_shape},\nlayout={self.layout})"
