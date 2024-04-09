import json
import os
from os.path import dirname, join

from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)
config = {
    "spark.driver.maxResultSize": os.environ.get("spark.driver.maxResultSize"),
    "spark.driver.memory": os.environ.get("spark.driver.memory"),
    "spark.executor.memory": os.environ.get("spark.executor.memory"),
    "spark.default.parallelism": os.environ.get("spark.default.parallelism"),
    "spark.sql.debug.maxToStringFields": os.environ.get("spark.sql.debug.maxToStringFields"),
    "extra_packages": json.loads(os.environ.get("extra_packages")),
    "s3.bucket.name": os.environ.get("s3.bucket.name"),
    "spark.hadoop.fs.s3a.aws.credentials.provider": os.environ.get("spark.hadoop.fs.s3a.aws.credentials.provider"),
    "spark.hadoop.fs.s3a.access.key": os.environ.get("aws_access_key_id"),
    "spark.hadoop.fs.s3a.secret.key": os.environ.get("aws_secret_access_key"),
    "spark.hadoop.fs.s3a.session.token": os.environ.get("aws_session_token"),
}
