import pytest
import jax.numpy as jnp

dtype = pytest.simple_fixture("dtype", params=(jnp.float64,))
