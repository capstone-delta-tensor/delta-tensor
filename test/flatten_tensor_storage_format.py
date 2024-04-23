import unittest
import numpy as np
import shutil

from api.delta_tensor import DeltaTensor
from util.spark_util import SparkUtil
from util.data_util import read_example_dense_tensor
from algorithms.flatten import get_first_last_chunk_ids


class TestTensor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dense = read_example_dense_tensor()
        cls.tensor_4d = np.stack([dense] * 24)
        cls.tensor_5d = np.stack([cls.tensor_4d] * 3)
        cls.slice_tuple_4d = ((0, 3), (0, 3), (0, 512), (0, 512))
        cls.slice_tuple_5d = ((0, 1), (0, 3), (0, 3), (0, 512), (0, 512))

    def setUp(self):
        self.delta_tensor = DeltaTensor(SparkUtil())

    def tearDown(self):
        self.delta_tensor.spark_util.stop_session();
        shutil.rmtree(SparkUtil.FTSF_TABLE, ignore_errors=True)

    def test_chunking(self):
        chunks_of_4dt4d = SparkUtil.flatten_to_chunks(self.tensor_4d, 4)
        self.assertEqual(len(chunks_of_4dt4d), 1)
        chunks_of_4dt3d = SparkUtil.flatten_to_chunks(self.tensor_4d, 3)
        self.assertEqual(len(chunks_of_4dt3d), self.tensor_4d.shape[0])
        chunks_of_4dt2d = SparkUtil.flatten_to_chunks(self.tensor_4d, 2)
        self.assertEqual(len(chunks_of_4dt2d), self.tensor_4d.shape[0] * self.tensor_4d.shape[1])
        chunks_of_5dt3d = SparkUtil.flatten_to_chunks(self.tensor_5d, 3)
        self.assertEqual(len(chunks_of_5dt3d), self.tensor_5d.shape[0] * self.tensor_5d.shape[1])

    def store_and_retrieve(self, tensor: np.ndarray, slice_tuple: tuple = (), chunk_dim_count: int = 3):
        t_id = self.delta_tensor.save_dense_tensor(tensor, chunk_dim_count)
        return self.delta_tensor.get_dense_tensor_by_id(t_id, slice_tuple)

    def check_write_read_equality(self, tensor: np.ndarray):
        self.assertTrue(np.array_equal(tensor, self.store_and_retrieve(tensor)))

    def test_4d(self):
        self.check_write_read_equality(self.tensor_4d)

    def test_5d(self):
        self.check_write_read_equality(self.tensor_5d)

    def test_get_sliced_tensor_chunk_range_1(self):
        chunked_tensor_shape = [24]
        self.assertEqual(get_first_last_chunk_ids(chunked_tensor_shape, self.slice_tuple_4d), (0, 2))

    def test_get_sliced_tensor_chunk_range_2(self):
        chunked_tensor_shape = [24, 3]
        self.assertEqual(get_first_last_chunk_ids(chunked_tensor_shape, self.slice_tuple_4d), (0, 8))

    def test_slicing_4d_tensor_3d_chunk(self):
        retrieved_sliced_tensor = self.store_and_retrieve(self.tensor_4d, self.slice_tuple_4d)
        expected_sliced_tensor = self.tensor_4d[0:3, :, :, :];
        self.assertTrue(np.array_equal(retrieved_sliced_tensor, expected_sliced_tensor))

    def test_slicing_4d_tensor_2d_chunk(self):
        retrieved_sliced_tensor = self.store_and_retrieve(self.tensor_4d, self.slice_tuple_4d, 2)
        expected_sliced_tensor = self.tensor_4d[0:3, :, :, :];
        self.assertTrue(np.array_equal(retrieved_sliced_tensor, expected_sliced_tensor))

    def test_slicing_5d_tensor_3d_chunk(self):
        retrieved_sliced_tensor = self.store_and_retrieve(self.tensor_5d, self.slice_tuple_5d)
        expected_sliced_tensor = self.tensor_5d[0:1, 0:3, :, :, :];
        self.assertTrue(np.array_equal(retrieved_sliced_tensor, expected_sliced_tensor))

    def test_slicing_5d_tensor_2d_chunk(self):
        retrieved_sliced_tensor = self.store_and_retrieve(self.tensor_5d, self.slice_tuple_5d, 2)
        expected_sliced_tensor = self.tensor_5d[0:1, 0:3, :, :, :];
        self.assertTrue(np.array_equal(retrieved_sliced_tensor, expected_sliced_tensor))


if __name__ == '__main__':
    unittest.main()
