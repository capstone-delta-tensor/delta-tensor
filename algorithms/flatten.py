import numpy as np

def get_first_last_chunk_ids(chunked_tensor_shape: list[int], slice_tuple: tuple):
    l_chunk_indices, r_chunk_indices = [], []
    for i, dim in enumerate(slice_tuple):
        if i >= len(chunked_tensor_shape):
            break
        l_chunk_indices.append(dim[0])
        r_chunk_indices.append(dim[1] - 1 if dim[1] > dim[0] else dim[0])
    
    l_chunk_id = np.ravel_multi_index(l_chunk_indices, chunked_tensor_shape)
    r_chunk_id = np.ravel_multi_index(r_chunk_indices, chunked_tensor_shape)

    return l_chunk_id, r_chunk_id
