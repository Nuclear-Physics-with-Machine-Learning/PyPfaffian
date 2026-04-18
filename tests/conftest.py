import pytest
import torch


def simple_fixture(name, params):
    @pytest.fixture(name=name, params=params)
    def inner(request):
        return request.param
    return inner

pytest.simple_fixture = simple_fixture

N     = pytest.simple_fixture("N", params=(2, 4, 6, 12, 16, 20))
seed  = pytest.simple_fixture("seed", params=(0, 42))
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
