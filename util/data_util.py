import os
import subprocess
import boto3
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from tensor.sparse_tensor import SparseTensorCOO
from settings import config
from boto3.s3.transfer import TransferConfig


def get_ffhq_location() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset', 'ffhq')


def read_png_as_tensor(path: str) -> np.ndarray:
    return iio.imread(path).transpose((2, 0, 1))


def read_example_dense_tensor() -> np.ndarray:
    return read_png_as_tensor(os.path.join(get_ffhq_location(), '00000.png'))


def read_ffhq_as_tensor(count: int) -> np.ndarray:
    img_tensors = []
    for i, file in enumerate(Path(get_ffhq_location()).iterdir()):
        if i >= count:
            break
        if not file.is_file():
            continue

        img_tensors.append(read_png_as_tensor(file))

    return np.stack(img_tensors)


def get_uber_dataset() -> SparseTensorCOO:
    uber_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset/uber/uber.tns')
    uber_sparse_tensor = np.loadtxt(uber_file_path, dtype=int).transpose()
    indices = uber_sparse_tensor[0:-1] - 1
    values = uber_sparse_tensor[-1].astype(float)
    dense_shape = (183, 24, 1140, 1717)
    return SparseTensorCOO(indices, values, dense_shape)


def get_s3_bucket() -> str:
    return config['s3.bucket.name']


def get_s3_location(key: str, bucket: str = get_s3_bucket()):
    return "/".join(("s3://" + bucket, key))


def get_size(path: str) -> str:
    if path.startswith('s3'):
        cmd = ['aws', 's3', 'ls', '--summarize', '--human-readable', '--recursive', path]
        output = subprocess.check_output(cmd).decode('utf-8').strip().split('\n')
        return output[-1]
    return subprocess.check_output(['du', '-sh', path]).split()[0].decode('utf-8')


def build_s3_client():
    return boto3.client("s3", aws_access_key_id=config["spark.hadoop.fs.s3a.access.key"],
                       aws_secret_access_key=config["spark.hadoop.fs.s3a.secret.key"],
                       aws_session_token=config["spark.hadoop.fs.s3a.session.token"]) 


def put_object_to_s3(byte_data: bytes, key: str, is_large: bool = False, bucket: str = get_s3_bucket()) -> None:
    if is_large:
        file_path = '/tmp/dense-tensor-binary'
        with open(file_path, 'wb') as f:
            f.write(byte_data)
        GB = 1024 ** 3
        config = TransferConfig(multipart_threshold=5*GB)
        build_s3_client().upload_file(file_path, bucket, key, Config=config)
        return
    
    build_s3_client().put_object(Body=byte_data, Bucket=bucket, Key=key)


def get_s3_object(key: str, bucket: str = get_s3_bucket()) -> bytes:
    return build_s3_client().get_object(Bucket=bucket, Key=key)['Body'].read()


def delete_s3_prefix(prefix: str) -> None:
    s3 = build_s3_client()
    response = s3.list_objects_v2(Bucket=get_s3_bucket(), Prefix=prefix)
    if 'Contents' not in response:
        return
    for object in response['Contents']:
            s3.delete_object(Bucket=get_s3_bucket(), Key=object['Key'])
