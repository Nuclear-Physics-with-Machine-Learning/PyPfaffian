# Python Pfaffian for GPUs

This repository contains a differentiable implementation of the Pfaffian operation for PyTorch, with optional GPU acceleration via Triton.  Implementations are based on the algorithms published by M. Wimmer: [https://arxiv.org/abs/1102.3440](https://arxiv.org/abs/1102.3440)

> What's the difference between this package and [pfapack](https://pypi.org/project/pfapack/)?  This package is implemented directly in PyTorch to enable GPU computations while pfapack is implemented with C / Fortran bindings for use in numpy on CPUs.  This package is also differentiable through the pfaffian operation.


## Installation

```bash
pip install torchpfaff
```

For GPU-accelerated Triton kernels on CUDA:

```bash
pip install torchpfaff[triton]
```

Alternatively, you can download the source code and install directly.

## Using this package

```python
from torchpfaff import pfaffian, log_pfaffian

# Construct an antisymmetric matrix:
M = ...

pf = pfaffian(M)

# For numerical stability with large matrices, use log-space:
sign, log_pf = log_pfaffian(M)
```


## Backends

### PyTorch
Full implementation with LTL (Parlett-Reid), direct, and recursive methods.  Analytical backward pass via `torch.autograd.Function`.  Supports `torch.vmap` and batched inputs.

### Triton (CUDA)
GPU kernel for the LTL decomposition on CUDA devices.  When Triton is installed and the input tensor is on CUDA, `log_pfaffian` automatically dispatches to the Triton kernel.  You can also force a specific backend:

```python
sign, log_pf = log_pfaffian(M, backend="triton")   # force Triton
sign, log_pf = log_pfaffian(M, backend="pytorch")  # force PyTorch
```
