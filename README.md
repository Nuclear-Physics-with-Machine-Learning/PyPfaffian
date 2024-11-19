# Python Pfaffian for GPUs

This repository contains implementations of the Pfaffian operation to be used in machine learning frameworks (Pytorch, JAX).  Implementations are based on the algorithms published by M. Wimmer: [https://arxiv.org/abs/1102.3440](https://arxiv.org/abs/1102.3440)

> What's the difference between this package and [pfapack](https://pypi.org/project/pfapack/)?  This package is implemented directly in the machine learning frameworks (JAX, Pytorch) to enable GPU computations while pfapack is implemented with C / Fortran bindings for use in numpy on CPUs.  This package is also differentiable through the pfaffian operation.


## Installation

To install this package, simple `pip install py-pfaffian`.  Alternatively, you can download the source code and install directly.

## Using this package

To call the pfaffian function from a framework, you can work like this:

```python
from py_pfaffian.jax import pfaffian

# Construct an antisymmetric matrix: 
M = ...

pf = pfaffian(M)
```


## Limitations

Currently, the pfaffian has implementations only in JAX, and only with the Parlett-Reid algorithm via decomposition.  The gradient computation is supported via a `custom_jvp` interface in JAX.  The algorithm is compatible with `jax.jit`, `jax.vmap`, and differentiation in JAX.

Pytorch has a basic implementation that may not be performant for larger matrices - and isn't vmapped at this time.

