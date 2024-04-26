# Delta Tensor

`deltatensor` enable users to use Delta Lake to manage their tensors.

## Development Environment Setup

### Install Nix

Follow the instructions [here](https://nixos.org/download/) to install `nix`. After the installation, add the below line to your `/etc/nix/nix.conf`

```
extra-experimental-features = nix-command flakes
```

Run the following command to enter a development shell environment

```bash
nix develop
```

### Install required packages

Create a python virtual environment (`Requires-Python >=3.7`)

```bash
python -m venv .py_env
source ./.py_env/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

If you are not using Nvidia GPUs, 

```bash
pip install -r requirements_no_cuda.txt
```

### Examples

#### General tensor
```python
from api.delta_tensor import *

tensor = np.zeros((24, 3, 1024, 1024), dtype=np.int8)
# Write a tensor
delta_tensor = DeltaTensor(SparkUtil())
t_id = delta_tensor.save_dense_tensor(tensor)
# Read a tensor
retrieved_tensor = delta_tensor.get_dense_tensor_by_id(tensor_id)
print(f"Data consistency {np.array_equal(retrieved_tensor, tensor)}")
# Read a slice of the tensor
slice_dim_start = 0
slice_dim_end = 5
slice = delta_tensor.get_dense_tensor_by_id(t_id, ((slice_dim_start, slice_dim_end), (0,3), (0,1024), (0,1024)))
print(f"Data consistency {np.array_equal(slice, tensor[0:5,:,:,:])}")
```
