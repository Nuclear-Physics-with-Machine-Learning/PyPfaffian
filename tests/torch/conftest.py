import pytest
import torch

dtype = pytest.simple_fixture("dtype", params=(torch.float32, torch.float64))


# Dtype-aware tolerance defaults.  Float32 LTL accumulates O(N^3) FMA
# rounding error so we keep an order of magnitude of slack vs. float64.
_TOL = {
    torch.float32: {"atol": 1e-4, "rtol": 1e-4},
    torch.float64: {"atol": 1e-8, "rtol": 1e-7},
}


@pytest.fixture
def tol(dtype):
    """Return (atol, rtol) tuned to the parametrized dtype."""
    t = _TOL[dtype]
    return t["atol"], t["rtol"]


@pytest.fixture
def skew_matrix(N, seed, dtype):
    """Generate a random skew-symmetric matrix."""
    torch.manual_seed(seed)
    M = torch.randn(N, N, dtype=dtype)
    return M - M.T


@pytest.fixture
def skew_matrix_batch(N, seed, dtype):
    """Generate a batch of random skew-symmetric matrices."""
    torch.manual_seed(seed)
    batch = 5
    M = torch.randn(batch, N, N, dtype=dtype)
    return M - M.transpose(-1, -2)
