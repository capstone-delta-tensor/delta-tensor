from api.delta_tensor import *
from util.spark_util import SparkUtil


def example_sparse_tensor(delta_tensor: DeltaTensor) -> None:
    indices = np.array([[0, 1, 1, 1],
                        [2, 0, 2, 1]])
    values = np.array([3, 4, 5, -1])
    dense_shape = (2, 4)
    sparse = SparseTensorCOO(indices, values, dense_shape)
    print("original tensor: ", sparse)
    t_id = delta_tensor.save_sparse_tensor(sparse, layout=SparseTensorLayout.CSR)
    print(t_id)
    tensor = delta_tensor.get_sparse_tensor_by_id(t_id, layout=SparseTensorLayout.CSR)
    print("restored tensor: ", tensor)

def test_coo(delta_tensor: DeltaTensor) -> None:
    indices = np.array([[0, 1, 1, 1],
                        [2, 0, 2, 1]])
    values = np.array([3, 4, 5, -1])
    dense_shape = (2, 4)
    tensor = SparseTensorCOO(indices, values, dense_shape)
    print("original tensor: ", tensor)
    coo = torch.sparse_coo_tensor(tensor.indices, tensor.values, tensor.dense_shape, dtype=torch.float32)
    print("coo tensor: ", coo)
    csr = coo.to_sparse_csr()
    print("csr tensor: ", csr)
    new_coo = csr.to_sparse_coo()
    print("restored coo: ", new_coo)
    indices = new_coo.indices().numpy()
    values = new_coo.values().numpy()    
    res_coo = SparseTensorCOO(indices, values, tensor.dense_shape)
    print("restored tensor: ", res_coo)


def benchmark_uber_dataset(delta_tensor: DeltaTensor) -> None:
    pass


if __name__ == '__main__':
    delta_tensor = DeltaTensor(SparkUtil())
    # Test for sparse tensor
    example_sparse_tensor(delta_tensor)

    # Test for uber set
    benchmark_uber_dataset(delta_tensor)
