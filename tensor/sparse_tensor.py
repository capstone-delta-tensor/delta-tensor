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
    
    def __eq__(self, other):
        if not isinstance(other, SparseTensorCOO):
            # don't attempt to compare against unrelated types
            return NotImplemented
        
        # Compare all attributes for equality. Use numpy.array_equal for comparing numpy arrays.
        return (np.array_equal(self.indices, other.indices) and
                np.array_equal(self.values, other.values) and
                self.dense_shape == other.dense_shape)

class SparseTensorCSR:
    def __init__(self, values: np.ndarray, col_indices: np.ndarray, crow_indices: np.ndarray, dense_shape: tuple):
        self.values = values
        self.col_indices = col_indices
        self.crow_indices = crow_indices
        self.dense_shape = dense_shape
        self.layout = SparseTensorLayout.CSR

    def __str__(self):
        return (f"SparseTensor(\nvalues=\n{self.values},\ncol_indices=\n{self.col_indices},\n"
                f"crow_indices=\n{self.crow_indices},\ndense_shape={self.dense_shape},\nlayout={self.layout})")


class SparseTensorCSC:
    def __init__(self, values: np.ndarray, row_indices: np.ndarray, ccol_indices: np.ndarray, dense_shape: tuple):
        self.values = values
        self.row_indices = row_indices
        self.ccol_indices = ccol_indices
        self.dense_shape = dense_shape
        self.layout = SparseTensorLayout.CSC

    def __str__(self):
        return (f"SparseTensor(\nvalues=\n{self.values},\nrow_indices=\n{self.row_indices},\n"
                f"ccol_indices=\n{self.ccol_indices},\ndense_shape={self.dense_shape},\nlayout={self.layout})")


class SparseTensorCSF:
    def __init__(self, fptrs: np.ndarray, fids: np.ndarray, values: np.ndarray, dense_shape: tuple):
        # TODO @kevinvan13 Add fields as needed
        self.fptrs = fptrs
        self.fids = fids
        self.values = values
        self.dense_shape = dense_shape
        self.layout = SparseTensorLayout.CSF

    def _format_subarray(self, array):
        """Formats a subarray to show the first three and last three elements."""
        if len(array) > 10:
            return f"[{', '.join(map(str, array[:3]))}, ..., {', '.join(map(str, array[-3:]))}]"
        else:
            return f"[{', '.join(map(str, array))}]"

    def _format_array(self, array):
        """Formats the array of arrays for pretty printing."""
        if isinstance(array, list) and all(isinstance(item, list) for item in array):  # List of lists
            formatted = "[\n    " + ",\n    ".join(self._format_subarray(subarray) for subarray in array) + "\n]"
        elif isinstance(array, list):  # Just a single list, for vals
            formatted = self._format_subarray(array)
        else:  # For the vals, which might not be a list of lists
            formatted = str(array)
        return formatted

    def __str__(self):
        formatted_fptrs = self._format_array(self.fptrs)
        formatted_fids = self._format_array(self.fids)
        formatted_vals = self._format_array(self.values)
        return f"SparseTensor(\nfptrs=\n{formatted_fptrs},\nfids=\n{formatted_fids},\nvalues=\n{formatted_vals},\ndense_shape={self.dense_shape},\nlayout={self.layout})"

    def __eq__(self, other):
        if not isinstance(other, SparseTensorCSF):
            return NotImplemented
        
        return (np.array_equal(self.fptrs, other.fptrs) and
                np.array_equal(self.fids, other.fids) and
                np.array_equal(self.values, other.values) and
                self.dense_shape == other.dense_shape)

    def expand_row(self, val_index):
        # Initialize the path with None values
        depth = len(self.fids)
        path = [None] * depth

        # Store the original value index for returning the correct value
        original_val_index = val_index
        
        # The value index in the bottom layer doesn't need conversion
        path[-1] = self.fids[-1][val_index]
        
        # Trace back through the fptrs to reconstruct the full index path
        for level in reversed(range(1, depth)):  # Start from the bottom level and work up
            # Use binary search to find the block in fptrs that contains val_index for this level
            left, right = 0, len(self.fptrs[level-1]) - 1
            if val_index >= self.fptrs[level-1][-1]:
                while level >= 1:
                    path[level-1] = self.fids[level-1][-1]
                    level -= 1
                break
            while left < right:
                mid = left + (right - left) // 2
                if self.fptrs[level-1][mid] <= val_index & val_index < self.fptrs[level-1][mid+1]:
                    path[level-1] = self.fids[level-1][mid]
                    val_index = mid
                    break
                elif self.fptrs[level-1][mid+1] <= val_index:
                    left = mid + 1
                else:
                    right = mid
            if left == right:
                path[level-1] = self.fids[level-1][left]
                val_index = left
       


        return path, self.values[original_val_index]

class SparseTensorModeGeneric:
    def __init__(self, indices: np.ndarray, values: np.ndarray, block_shape: tuple, dense_shape: tuple):
        self.indices = indices
        self.values = values
        self.block_shape = block_shape
        self.dense_shape = dense_shape
        self.layout = SparseTensorLayout.MODE_GENERIC

    def __str__(self):
        return f"SparseTensor(\nindices=\n{self.indices},\nvalues=\n{self.values},\nblock_shape={self.block_shape},\ndense_shape={self.dense_shape},\nlayout={self.layout})"

    