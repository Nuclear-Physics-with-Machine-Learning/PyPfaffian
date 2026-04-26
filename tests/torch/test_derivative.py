import pytest

import torch

from torchpfaff.pfaffian import LogPfaffian
from torchpfaff.utils import pfaffian_LTL


def test_log_pfaffian_gradcheck(N, seed, dtype):
    """Verify LogPfaffian backward against numerical finite differences.

    Parametrize through antisymmetrization so gradcheck's per-element
    perturbations never break skew-symmetry.
    """
    torch.manual_seed(seed)
    M = torch.randn(N, N, dtype=dtype, requires_grad=True)

    def logpf_via_skew(M):
        A = M - M.T
        _sign, logpf = LogPfaffian.apply(A)
        return logpf

    assert torch.autograd.gradcheck(logpf_via_skew, (M,), atol=1e-5, rtol=1e-4)


def test_log_pfaffian_vjp_analytical(skew_matrix):
    """Gradient of log|pf(A)| w.r.t. A should equal -1/2 * A^{-1}."""
    A = skew_matrix.clone().requires_grad_(True)

    _sign, logpf = LogPfaffian.apply(A)
    logpf.backward()

    expected = -0.5 * torch.linalg.inv(skew_matrix)
    assert torch.allclose(A.grad, expected, atol=1e-8), \
        f"max error: {(A.grad - expected).abs().max().item()}"


def test_pfaffian_jvp_numerical(N, seed, dtype):
    """Directional derivative of pf(A) via backward matches central finite differences."""
    if N < 4:
        pytest.skip("Tangent perturbs entry [0,3]; need N >= 4")

    torch.manual_seed(int(seed))
    A = torch.randn(N, N, dtype=dtype)
    A = A - A.T

    # Skew-symmetric tangent touching [0,3] and [3,0]
    tangent = torch.zeros_like(A)
    tangent[0, 3] = 1.0
    tangent = tangent - tangent.T

    # Analytical directional derivative via backward
    A_diff = A.clone().requires_grad_(True)
    sign, logpf = LogPfaffian.apply(A_diff)
    pf = sign * torch.exp(logpf)
    pf.backward()
    analytical_jvp = (A_diff.grad * tangent).sum()

    # Central finite-difference
    kick = 1e-5
    pf_plus  = pfaffian_LTL(A + 0.5 * kick * tangent)
    pf_minus = pfaffian_LTL(A - 0.5 * kick * tangent)
    numerical_jvp = (pf_plus - pf_minus) / kick

    assert torch.allclose(analytical_jvp, numerical_jvp, atol=1e-4), \
        f"analytical={analytical_jvp.item()}, numerical={numerical_jvp.item()}"


def test_log_pfaffian_vmap(skew_matrix_batch):
    """vmap over LogPfaffian.apply matches per-matrix calls."""
    batch = skew_matrix_batch.shape[0]

    sign_vmap, logpf_vmap = torch.vmap(LogPfaffian.apply)(skew_matrix_batch)

    for i in range(batch):
        sign_i, logpf_i = LogPfaffian.apply(skew_matrix_batch[i])
        assert torch.allclose(sign_vmap[i], sign_i), \
            f"batch[{i}] sign: vmap={sign_vmap[i].item()}, single={sign_i.item()}"
        assert torch.allclose(logpf_vmap[i], logpf_i, atol=1e-10), \
            f"batch[{i}] logpf: vmap={logpf_vmap[i].item()}, single={logpf_i.item()}"
