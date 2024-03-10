import numpy as np


class SparseTensorBSR:
    def __init__(self, indices: np.ndarray, values: np.ndarray, block_shape: tuple, dense_shape: tuple):
        self.indices = indices
        self.values = values
        self.block_shape = block_shape
        self.dense_shape = dense_shape

    def __str__(self):
        return f"SparseTensor(\nindices=\n{self.indices},\nvalues=\n{self.values},\nblock_shape={self.block_shape},\ndense_shape={self.dense_shape})"
