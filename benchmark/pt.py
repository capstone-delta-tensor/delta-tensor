import statistics
import time

import torch
from botocore.client import BaseClient

from tensor.sparse_tensor import SparseTensorCOO
from util.data_util import get_uber_dataset, build_s3_client
from util.spark_util import SparkUtil


def save_tensor(obj: BaseClient, tensor: torch.Tensor, local_path: str, bucket: str, file_key: str) -> None:
    torch.save(tensor, local_path)
    obj.upload_file(Filename=local_path, Bucket=bucket, Key=file_key)


def load_tensor(obj: BaseClient, local_path: str, bucket: str, file_key: str) -> torch.Tensor:
    obj.download_file(Filename=local_path, Bucket=bucket, Key=file_key)
    return torch.load(local_path)


def benchmark_uber_dataset_as_pt(obj: BaseClient, sparse: SparseTensorCOO) -> tuple[float, float, float]:
    tensor = torch.sparse_coo_tensor(torch.tensor(sparse.indices), torch.tensor(sparse.values), sparse.dense_shape)
    local_pt_path = '/tmp/tensor.pt'
    s3_bucket_name = SparkUtil.BUCKET.split('/')[-1]
    s3_pt_key = 'pytorch/tensor.pt'

    start = time.time()
    save_tensor(obj, tensor, local_pt_path, s3_bucket_name, s3_pt_key)
    pt_saving_time = time.time() - start
    print(f"Tensor saving time: {pt_saving_time} seconds")

    start = time.time()
    load_tensor(obj, local_pt_path, s3_bucket_name, s3_pt_key)
    pt_loading_time = time.time() - start
    print(f"Tensor loading time: {pt_loading_time} seconds")

    start = time.time()
    _ = load_tensor(obj, local_pt_path, s3_bucket_name, s3_pt_key)[0]
    slicing_time = time.time() - start
    print(f"Tensor slicing time: {slicing_time} seconds")
    return pt_saving_time, pt_loading_time, slicing_time


if __name__ == '__main__':
    obj = build_s3_client()

    # Load uber dataset
    uber_sparse = get_uber_dataset()

    # Epoch number
    epoch = 10

    # Test for torch pt file
    stats = [benchmark_uber_dataset_as_pt(obj, uber_sparse) for _ in range(epoch)]

    # Display statistics
    print(f"=====Uber dataset benchmark results for PyTorch pt file format=====")
    print(f"Average tensor insertion time for {epoch} epochs: {statistics.mean([_[0] for _ in stats])}")
    print(f"Average tensor full scan time for {epoch} epochs: {statistics.mean([_[1] for _ in stats])}")
    print(f"Average tensor slicing time for {epoch} epochs: {statistics.mean([_[2] for _ in stats])}")
