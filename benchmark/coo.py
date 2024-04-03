import torch

from api.delta_tensor import *
import numpy as np
import time
from api.delta_tensor import SparseTensorCOO

def benchmark_uber_dataset(delta_tensor: DeltaTensor) -> None:
    print("=======================================")
    print("COO for uber dataset")
    uber_sparse_tensor = np.loadtxt("/home/danny/delta-tensor/dataset/uber/uber.tns", dtype=int).transpose()
    indices = uber_sparse_tensor[0:-1]
    values = uber_sparse_tensor[-1]
    dense_shape = (183, 24, 1140, 1717)
    sparse = SparseTensorCOO(indices, values, dense_shape)
    start = time.time()
    t_id = delta_tensor.save_sparse_tensor(sparse, layout=SparseTensorLayout.COO, block_shape=(4, 4))
    print(f"Tensor insertion time: {time.time() - start} seconds")
    start = time.time()
    retrieved = delta_tensor.get_sparse_tensor_by_id(t_id, layout=SparseTensorLayout.COO)
    print(f"Tensor retrieving time: {time.time() - start} seconds")
    print(sparse == retrieved)



if __name__ == '__main__':
    delta_tensor = DeltaTensor(SparkUtil())

    # Test for uber set
    benchmark_uber_dataset(delta_tensor)
