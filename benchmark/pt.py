import time

import boto3
import torch
from botocore.client import BaseClient

from tensor.sparse_tensor import SparseTensorCOO
from util.data_util import get_uber_dataset
from util.spark_util import SparkUtil


def benchmark_uber_dataset_as_pt(obj: BaseClient, sparse: SparseTensorCOO) -> None:
    tensor = torch.sparse_coo_tensor(torch.tensor(sparse.indices), torch.tensor(sparse.values), sparse.dense_shape)
    local_pt_path = '/tmp/pytorch/tensor.pt'
    s3_bucket_name = SparkUtil.BUCKET.split('/')[-1]
    s3_pt_key = 'pytorch/tensor.pt'
    start = time.time()
    torch.save(tensor, local_pt_path)
    obj.upload_file(Filename=local_pt_path, Bucket=s3_bucket_name, Key=s3_pt_key)
    print(f"Tensor saving time: {time.time() - start} seconds")
    start = time.time()
    obj.download_file(Filename=local_pt_path, Bucket=s3_bucket_name, Key=s3_pt_key)
    torch.load(local_pt_path)
    print(f"Tensor loading time: {time.time() - start} seconds")


if __name__ == '__main__':
    obj = boto3.client("s3")

    # Load uber dataset
    uber_sparse = get_uber_dataset()

    # Test for torch pt file
    benchmark_uber_dataset_as_pt(obj, uber_sparse)
