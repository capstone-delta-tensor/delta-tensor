from delta_tensor import *
from spark_util import SparkUtil

if __name__ == '__main__':
    dense = np.zeros([3, 2, 2, 3, 2])
    dense[0, 0, 0, :, :] = np.arange(6).reshape(3, 2)
    dense[1, 1, 0, :, :] = np.arange(6, 12).reshape(3, 2)
    dense[2, 1, 1, :, :] = np.arange(12, 18).reshape(3, 2)

    spark_util = SparkUtil()

    t_id = insert_tensor(spark_util, dense)
    print(t_id)
    tensor = find_tensor_by_id(spark_util, t_id)
    print(tensor)
    print(np.array_equal(tensor, dense))
