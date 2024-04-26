import shutil
import subprocess

from api.delta_tensor import *
from util.data_util import read_ffhq_as_tensor
from random import randint

NUMBER_OF_IMG = 10000
SLICE_SIZE = 100
DENSE_TENSOR_BINARY_LOCATION = '/tmp/dense_tensor_binary'


def get_size(path: str) -> str:
    return subprocess.check_output(['du', '-sh', path]).split()[0].decode('utf-8')


def get_first_tensor_id() -> str:
    delta_tensor = DeltaTensor(SparkUtil())
    df = delta_tensor.spark_util.spark.read.format("delta").load(SparkUtil.FTSF_TABLE)
    tensor_id = df.select("id").first()['id']
    delta_tensor.spark_util.stop_session()
    return tensor_id


def benchmark_direct_serialization() -> None:
    print("=======================================")
    print("Direct serialization test for the ffhq dataset")
    tensor = read_ffhq_as_tensor(NUMBER_OF_IMG)

    start = time.time()
    with open(DENSE_TENSOR_BINARY_LOCATION, 'wb') as f:
        np.save(f, tensor)
    print(f"Bulk write time: {time.time() - start} seconds")

    print(f"tensor storage size: {get_size(DENSE_TENSOR_BINARY_LOCATION)}")


def benchmark_direct_deserialization() -> np.ndarray:
    print("=======================================")
    print("Direct deserialization test for the ffhq dataset")

    start = time.time()
    with open(DENSE_TENSOR_BINARY_LOCATION, 'rb') as f:
        deserialized_tensor = np.load(f)
    print(f"Bulk read time: {time.time() - start} seconds")
    return deserialized_tensor


def benchmark_ffhq_bulk_write() -> str:
    print("=======================================")
    print("FTSF bulk write test for the ffhq dataset")
    tensor = read_ffhq_as_tensor(NUMBER_OF_IMG)
    delta_tensor = DeltaTensor(SparkUtil())

    start = time.time()
    t_id = delta_tensor.save_dense_tensor(tensor)
    print(f"Bulk write time: {time.time() - start} seconds")

    print(f"tensor storage size: {get_size(SparkUtil.FTSF_TABLE)}")
    delta_tensor.spark_util.stop_session()
    return t_id


def benchmark_ffhq_bulk_read(tensor_id: str) -> np.ndarray:
    print("=======================================")
    print("FTSF bulk read test for the ffhq dataset")
    delta_tensor = DeltaTensor(SparkUtil())

    start = time.time()
    tensor = delta_tensor.get_dense_tensor_by_id(tensor_id)
    print(f"Bulk read time: {time.time() - start} seconds")
    delta_tensor.spark_util.stop_session()
    return tensor


def benchmark_direct_deserialization_and_slice(slice_dim_start: int, slice_dim_end: int) -> np.ndarray:
    print("=======================================")
    print("Direct deserialization test for the ffhq dataset")

    start = time.time()
    with open(DENSE_TENSOR_BINARY_LOCATION, 'rb') as f:
        deserialized_tensor = np.load(f)[slice_dim_start:slice_dim_end, :]
    print(f"Slice read time: {time.time() - start} seconds")
    return deserialized_tensor


def benchmark_ffhq_part_read(tensor_id: str, slice_dim_start: int, slice_dim_end: int) -> np.ndarray:
    print("=======================================")
    print("FTSF part read test for the ffhq dataset")
    delta_tensor = DeltaTensor(SparkUtil())

    start = time.time()
    tensor = delta_tensor.get_dense_tensor_by_id(tensor_id,
                                                 ((slice_dim_start, slice_dim_end), (0, 3), (0, 512), (0, 512)))
    print(f"Slice read time: {time.time() - start} seconds")
    delta_tensor.spark_util.stop_session()
    return tensor


if __name__ == '__main__':
    benchmark_direct_serialization()
    bulk_direct = benchmark_direct_deserialization()

    shutil.rmtree(SparkUtil.FTSF_TABLE, ignore_errors=True)
    benchmark_ffhq_bulk_write()
    t_id = get_first_tensor_id()
    bulk_ftsf = benchmark_ffhq_bulk_read(t_id)

    print(f"Data consistency {np.array_equal(bulk_direct, bulk_ftsf)}")

    slice_dim_start = randint(0, NUMBER_OF_IMG - SLICE_SIZE)
    slice_dim_end = slice_dim_start + SLICE_SIZE
    slice_direct = benchmark_direct_deserialization_and_slice(slice_dim_start, slice_dim_end)
    slice_ftsf = benchmark_ffhq_part_read(t_id, slice_dim_start, slice_dim_end)
    print(f"Data consistency {np.array_equal(slice_direct, slice_ftsf)}")
