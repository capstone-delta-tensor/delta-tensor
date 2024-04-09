import shutil
import subprocess

from api.delta_tensor import *
from util.data_util import read_ffhq_as_tensor

NUMBER_OF_IMG = 10000
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


def benchmark_direct_deserialization() -> None:
    print("=======================================")
    print("Direct deserialization test for the ffhq dataset")

    start = time.time()
    with open(DENSE_TENSOR_BINARY_LOCATION, 'rb') as f:
        deserialized_tensor = np.load(f)
    print(f"Bulk read time: {time.time() - start} seconds")


def benchmark_ffhq_bulk_write() -> str:
    print("=======================================")
    print("FSSF Bulk write test for the ffhq dataset")
    tensor = read_ffhq_as_tensor(NUMBER_OF_IMG)
    delta_tensor = DeltaTensor(SparkUtil())

    start = time.time()
    t_id = delta_tensor.save_dense_tensor(tensor)
    print(f"Bulk write time: {time.time() - start} seconds")

    print(f"tensor storage size: {get_size(SparkUtil.FTSF_TABLE)}")
    delta_tensor.spark_util.stop_session()
    return t_id


def benchmark_ffhq_bulk_read(tensor_id: str) -> None:
    print("=======================================")
    print("FSSF Bulk read test for the ffhq dataset")
    delta_tensor = DeltaTensor(SparkUtil())

    start = time.time()
    delta_tensor.get_dense_tensor_by_id(tensor_id)
    print(f"Bulk read time: {time.time() - start} seconds")
    delta_tensor.spark_util.stop_session()


if __name__ == '__main__':
    benchmark_direct_serialization()
    benchmark_direct_deserialization()

    shutil.rmtree(SparkUtil.FTSF_TABLE, ignore_errors=True)
    benchmark_ffhq_bulk_write()
    t_id = get_first_tensor_id()
    benchmark_ffhq_bulk_read(t_id)
