import numpy as np

from api.delta_tensor import DeltaTensor
from util.spark_util import SparkUtil
from util.data_util import read_example_dense_tensor

delta_tensor = DeltaTensor(SparkUtil())

dense = read_example_dense_tensor()
tensor = np.stack([dense]*24)
t_id = delta_tensor.save_dense_tensor(tensor)
restored_tensor = delta_tensor.get_dense_tensor_by_id(t_id)