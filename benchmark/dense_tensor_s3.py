import io
import tempfile

from api.delta_tensor import *
from util.data_util import *
from random import randint

NUMBER_OF_IMG = 10000
SLICE_SIZE = 100
S3_BINARY_KEY = 'tmp/dense_tensor_binary'
S3_FTSF_PREFIX = 'flattened'


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

    fd, path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, 'w') as tmp:
            np.save(tmp, tensor)
            GB = 1024 ** 3
            config = TransferConfig(multipart_threshold=5*GB)
            start = time.time()
            build_s3_client().upload_file(tmp, get_s3_bucket, S3_BINARY_KEY, Config=config)
            print(f"Bulk write time: {time.time() - start} seconds")
    finally:
        os.remove(path)

    print(f"tensor storage size: {get_size(get_s3_location(S3_BINARY_KEY))}")


def benchmark_direct_deserialization() -> np.ndarray:
    print("=======================================")
    print("Direct deserialization test for the ffhq dataset")

    start = time.time()
    deserialized_tensor = np.load(io.BytesIO(get_s3_object(S3_BINARY_KEY)))
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

    print(f"tensor storage size: {get_size(get_s3_location(S3_FTSF_PREFIX))}")
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
    print("Direct deserialization part read test for the ffhq dataset")

    start = time.time()
    deserialized_tensor = np.load(io.BytesIO(get_s3_object(S3_BINARY_KEY)))[slice_dim_start:slice_dim_end, :]
    print(f"Slice read time: {time.time() - start} seconds")
    return deserialized_tensor


def benchmark_ffhq_part_read(tensor_id: str, slice_dim_start: int, slice_dim_end: int) -> np.ndarray:
    print("=======================================")
    print("FTSF part read test for the ffhq dataset")
    delta_tensor = DeltaTensor(SparkUtil())

    start = time.time()
    tensor = delta_tensor.get_dense_tensor_by_id(tensor_id, ((slice_dim_start, slice_dim_end), (0,3), (0,512), (0,512)))
    print(f"Slice read time: {time.time() - start} seconds")
    delta_tensor.spark_util.stop_session()
    return tensor


if __name__ == '__main__':
    delete_s3_prefix(S3_BINARY_KEY)
    delete_s3_prefix(S3_FTSF_PREFIX)
    benchmark_direct_serialization()
    bulk_direct = benchmark_direct_deserialization()

    benchmark_ffhq_bulk_write()
    t_id = get_first_tensor_id()
    bulk_ftsf = benchmark_ffhq_bulk_read(t_id)

    print(f"Data consistency {np.array_equal(bulk_direct, bulk_ftsf)}")


    slice_dim_start = randint(0, NUMBER_OF_IMG - SLICE_SIZE)
    slice_dim_end = slice_dim_start + SLICE_SIZE
    slice_direct = benchmark_direct_deserialization_and_slice(slice_dim_start, slice_dim_end)
    slice_ftsf = benchmark_ffhq_part_read(t_id, slice_dim_start, slice_dim_end)
    print(f"Data consistency {np.array_equal(slice_direct, slice_ftsf)}")
