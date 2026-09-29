# Installation

```bash
pip install cpd-pytorch
```

This installs NumPy and PyTorch if they are missing. Add matplotlib, needed by the
example scripts, with:

```bash
pip install "cpd-pytorch[examples]"
```

## GPU support

CPD-PyTorch runs on any device PyTorch supports. pip installs the default PyTorch build
for your platform: CUDA-enabled on Linux, CPU-only on Windows, CPU and MPS on macOS. For a
specific CUDA version, install PyTorch first with the command from
[pytorch.org](https://pytorch.org/get-started/locally/), then install CPD-PyTorch.

Check what is available:

```python
import torch

print(torch.cuda.is_available())  # NVIDIA GPU
print(torch.backends.mps.is_available())  # Apple Silicon GPU
```

## Supported versions

| Dependency | Versions |
|---|---|
| Python | 3.8 - 3.14 |
| PyTorch | 1.10 or newer |
| NumPy | 1.21 or newer |

CI runs the tests on every supported Python version, on Linux, macOS and Windows, and
with the oldest supported combination (Python 3.8, PyTorch 1.10.2, NumPy 1.21.6).

!!! note "Older PyTorch releases"
    PyTorch wheels older than 2.3 were built against NumPy 1.x: install `"numpy<2"` with
    them. PyTorch 1.x also imports `pkg_resources` at start-up, which needs
    `"setuptools<70"`.

## From source

```bash
pip install git+https://github.com/YuliangXiaoYLX/CPD-Pytorch.git
```

For development (tests, linting, documentation) see
[Contributing](contributing.md).
