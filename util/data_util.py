import os
import imageio
import numpy as np

def read_png_as_tensor(example_path: str) -> np.ndarray:
    return imageio.imread(example_path).transpose((2, 0, 1))

def read_example_dense_tensor():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    example_path = os.path.join(current_dir, '..', 'dataset', 'ffhq', '69000', '69999.png')
    return read_png_as_tensor(example_path)
