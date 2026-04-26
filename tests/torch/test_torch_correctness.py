import pytest
import torch

from torchpfaff import log_pfaffian, LogPfaffian
from torchpfaff.utils import pfaffian_recursive, pfaffian_LTL, batched_log_pfaffian_ltl


def test_pfaffian_ltl_matches_recursive(skew_matrix):
    """LTL Pfaffian matches the recursive reference implementation."""
    N = skew_matrix.shape[0]
    if N > 10:
        pytest.skip("Recursive is O(N!) -- skip for large N")

    pf_ltl = pfaffian_LTL(skew_matrix)
    pf_rec = pfaffian_recursive(skew_matrix)
    assert torch.allclose(pf_ltl, pf_rec, atol=1e-8), \
        f"LTL={pf_ltl.item()}, recursive={pf_rec.item()}"


def test_log_pfaffian_reconstructs_pfaffian(skew_matrix):
    """sign * exp(log|pf|) should equal the Pfaffian value."""
    sign, logpf = log_pfaffian(skew_matrix, backend="pytorch")
    pf_from_log = sign * torch.exp(logpf)
    pf_ltl = pfaffian_LTL(skew_matrix)
    assert torch.allclose(pf_from_log, pf_ltl, atol=1e-6), \
        f"reconstructed={pf_from_log.item()}, LTL={pf_ltl.item()}"


def test_pf_squared_equals_det(skew_matrix):
    """Fundamental identity: pf(A)^2 = det(A)."""
    sign, logpf = log_pfaffian(skew_matrix, backend="pytorch")
    pf = sign * torch.exp(logpf)
    det = torch.linalg.det(skew_matrix)
    assert torch.allclose(pf**2, det, atol=1e-4), \
        f"pf^2={( pf**2).item()}, det={det.item()}"


def test_sign_flips_under_particle_swap(skew_matrix):
    """Swapping two rows+cols of A should flip the sign of pf(A)."""
    N = skew_matrix.shape[0]
    if N < 4:
        pytest.skip("Need at least 4 particles for a meaningful swap test")

    sign_orig, logpf_orig = log_pfaffian(skew_matrix, backend="pytorch")

    # Swap rows/cols 0 and 1
    P = torch.eye(N, dtype=skew_matrix.dtype)
    P[0, 0] = 0; P[1, 1] = 0; P[0, 1] = 1; P[1, 0] = 1
    A_swapped = P @ skew_matrix @ P.T

    sign_swapped, logpf_swapped = log_pfaffian(A_swapped, backend="pytorch")

    # log|pf| should be unchanged, sign should flip
    assert torch.allclose(logpf_orig, logpf_swapped, atol=1e-8), \
        f"logpf changed: {logpf_orig.item()} vs {logpf_swapped.item()}"
    assert torch.allclose(sign_orig, -sign_swapped, atol=1e-8), \
        f"sign didn't flip: {sign_orig.item()} vs {sign_swapped.item()}"


def test_odd_dimension_zero(seed, dtype):
    """Pfaffian of an odd-dimensional skew-symmetric matrix is zero."""
    for N in [3, 5, 7]:
        torch.manual_seed(seed)
        A = torch.randn(N, N, dtype=dtype)
        A = A - A.T
        pf = pfaffian_LTL(A)
        assert pf.item() == 0.0, f"pf({N}x{N}) = {pf.item()}, expected 0"


def test_batched_matches_individual(skew_matrix_batch):
    """Batched log_pfaffian should match per-matrix computation."""
    batch = skew_matrix_batch.shape[0]
    sign_b, logpf_b = log_pfaffian(skew_matrix_batch, backend="pytorch")

    for i in range(batch):
        sign_i, logpf_i = log_pfaffian(skew_matrix_batch[i], backend="pytorch")
        assert torch.allclose(sign_b[i], sign_i), \
            f"batch[{i}] sign mismatch: {sign_b[i].item()} vs {sign_i.item()}"
        assert torch.allclose(logpf_b[i], logpf_i, atol=1e-10), \
            f"batch[{i}] logpf mismatch: {logpf_b[i].item()} vs {logpf_i.item()}"


def test_batched_pf_squared_equals_det(skew_matrix_batch):
    """pf(A)^2 = det(A) for each matrix in a batch."""
    sign_b, logpf_b = log_pfaffian(skew_matrix_batch, backend="pytorch")
    pf_b = sign_b * torch.exp(logpf_b)
    det_b = torch.linalg.det(skew_matrix_batch)
    assert torch.allclose(pf_b**2, det_b, atol=1e-4)
