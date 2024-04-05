import os
import numpy as np

from pathlib import Path

import imageio.v3 as iio

def get_ffhq_location() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset', 'ffhq')

def read_png_as_tensor(path: str) -> np.ndarray:
    return iio.imread(path).transpose((2, 0, 1))

def read_example_dense_tensor() -> np.ndarray:
    return read_png_as_tensor(os.path.join(get_ffhq_location(), '00000.png'))

def read_ffhq_as_tensor(count: int) -> np.ndarray:
    img_tensors = []
    for i, file in  enumerate(Path(get_ffhq_location()).iterdir()):
        if i >= count:
            break
        if not file.is_file():
            continue

        img_tensors.append(read_png_as_tensor(file))
    
    return np.stack(img_tensors)
