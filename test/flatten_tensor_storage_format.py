import unittest
import numpy as np
import shutil

from api.delta_tensor import DeltaTensor
from util.spark_util import SparkUtil
from util.data_util import read_example_dense_tensor

class TestTensor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dense = read_example_dense_tensor()
        cls.tensor_4d = np.stack([dense]*24)
        cls.tensor_5d = np.stack([cls.tensor_4d]*3)
        cls.delta_tensor = DeltaTensor(SparkUtil())
    
    def tearDown(self):
        self.delta_tensor.spark_util.clear_cache();
        shutil.rmtree(SparkUtil.FTSF_LOCATION_FS, ignore_errors=True)

    def test_chunking(self):
        chunks_of_4dt4d = SparkUtil.flatten_to_chunks(self.tensor_4d, 4)
        self.assertEqual(len(chunks_of_4dt4d), 1)
        chunks_of_4dt3d = SparkUtil.flatten_to_chunks(self.tensor_4d, 3)
        self.assertEqual(len(chunks_of_4dt3d), self.tensor_4d.shape[0])
        chunks_of_4dt2d = SparkUtil.flatten_to_chunks(self.tensor_4d, 2)
        self.assertEqual(len(chunks_of_4dt2d), self.tensor_4d.shape[0] * self.tensor_4d.shape[1])
        chunks_of_5dt3d = SparkUtil.flatten_to_chunks(self.tensor_5d, 3)
        self.assertEqual(len(chunks_of_5dt3d), self.tensor_5d.shape[0] * self.tensor_5d.shape[1])
    
    def check_write_read_equality(self, tensor: np.ndarray):
        t_id = self.delta_tensor.save_dense_tensor(tensor)
        restored_tensor = self.delta_tensor.get_dense_tensor_by_id(t_id)
        self.assertEqual(True, np.array_equal(tensor,restored_tensor))

    def test_4d(self):
        self.check_write_read_equality(self.tensor_4d)
    
    def test_5d(self):
        self.check_write_read_equality(self.tensor_5d)

if __name__ == '__main__':
    unittest.main()
