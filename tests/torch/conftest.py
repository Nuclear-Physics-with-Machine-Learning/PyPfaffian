import pytest
import torch


dtype = pytest.simple_fixture("dtype", params=(torch.float64,))


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
