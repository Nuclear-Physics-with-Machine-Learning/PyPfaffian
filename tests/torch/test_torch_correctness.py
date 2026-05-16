import pytest
import torch

from torchpfaff import log_pfaffian, pfaffian
from torchpfaff.utils import pfaffian_direct, pfaffian_LTL, pfaffian_recursive


def test_pfaffian_ltl_matches_recursive(skew_matrix, tol):
    """LTL Pfaffian matches the recursive reference implementation."""
    N = skew_matrix.shape[0]
    if N > 10:
        pytest.skip("Recursive is O(N!) -- skip for large N")

    atol, rtol = tol
    pf_ltl = pfaffian_LTL(skew_matrix)
    pf_rec = pfaffian_recursive(skew_matrix)
    assert torch.allclose(pf_ltl, pf_rec, atol=atol, rtol=rtol), (
        f"LTL={pf_ltl.item()}, recursive={pf_rec.item()}"
    )


def test_log_pfaffian_reconstructs_pfaffian(skew_matrix, tol):
    """sign * exp(log|pf|) should equal the Pfaffian value."""
    atol, rtol = tol
    sign, logpf = log_pfaffian(skew_matrix, backend="pytorch")
    pf_from_log = sign * torch.exp(logpf)
    pf_ltl = pfaffian_LTL(skew_matrix)
    assert torch.allclose(pf_from_log, pf_ltl, atol=atol, rtol=rtol), (
        f"reconstructed={pf_from_log.item()}, LTL={pf_ltl.item()}"
    )


def test_pf_squared_equals_det(skew_matrix, tol):
    """Fundamental identity: pf(A)^2 = det(A)."""
    atol, rtol = tol
    sign, logpf = log_pfaffian(skew_matrix, backend="pytorch")
    pf = sign * torch.exp(logpf)
    det = torch.linalg.det(skew_matrix)
    # det involves an O(N^3) cancellation cascade so loosen tolerances.
    assert torch.allclose(pf**2, det, atol=max(atol, 1e-4), rtol=max(rtol, 1e-4)), (
        f"pf^2={(pf**2).item()}, det={det.item()}"
    )


def test_sign_flips_under_particle_swap(skew_matrix, tol):
    """Swapping two rows+cols of A should flip the sign of pf(A)."""
    N = skew_matrix.shape[0]
    if N < 4:
        pytest.skip("Need at least 4 particles for a meaningful swap test")

    atol, rtol = tol
    sign_orig, logpf_orig = log_pfaffian(skew_matrix, backend="pytorch")

    # Swap rows/cols 0 and 1
    P = torch.eye(N, dtype=skew_matrix.dtype)
    P[0, 0] = 0
    P[1, 1] = 0
    P[0, 1] = 1
    P[1, 0] = 1
    A_swapped = P @ skew_matrix @ P.T

    sign_swapped, logpf_swapped = log_pfaffian(A_swapped, backend="pytorch")

    # log|pf| should be unchanged, sign should flip
    assert torch.allclose(logpf_orig, logpf_swapped, atol=atol, rtol=rtol), (
        f"logpf changed: {logpf_orig.item()} vs {logpf_swapped.item()}"
    )
    assert torch.allclose(sign_orig, -sign_swapped, atol=atol, rtol=rtol), (
        f"sign didn't flip: {sign_orig.item()} vs {sign_swapped.item()}"
    )


def test_odd_dimension_zero(seed, dtype):
    """Pfaffian of an odd-dimensional skew-symmetric matrix is zero."""
    for N in [3, 5, 7]:
        torch.manual_seed(seed)
        A = torch.randn(N, N, dtype=dtype)
        A = A - A.T
        pf = pfaffian_LTL(A)
        assert pf.item() == 0.0, f"pf({N}x{N}) = {pf.item()}, expected 0"


def test_batched_matches_individual(skew_matrix_batch, tol):
    """Batched log_pfaffian should match per-matrix computation."""
    atol, rtol = tol
    batch = skew_matrix_batch.shape[0]
    sign_b, logpf_b = log_pfaffian(skew_matrix_batch, backend="pytorch")

    for i in range(batch):
        sign_i, logpf_i = log_pfaffian(skew_matrix_batch[i], backend="pytorch")
        assert torch.allclose(sign_b[i], sign_i), (
            f"batch[{i}] sign mismatch: {sign_b[i].item()} vs {sign_i.item()}"
        )
        assert torch.allclose(logpf_b[i], logpf_i, atol=atol, rtol=rtol), (
            f"batch[{i}] logpf mismatch: {logpf_b[i].item()} vs {logpf_i.item()}"
        )


def test_batched_pf_squared_equals_det(skew_matrix_batch, tol):
    """pf(A)^2 = det(A) for each matrix in a batch."""
    atol, rtol = tol
    sign_b, logpf_b = log_pfaffian(skew_matrix_batch, backend="pytorch")
    pf_b = sign_b * torch.exp(logpf_b)
    det_b = torch.linalg.det(skew_matrix_batch)
    assert torch.allclose(pf_b**2, det_b, atol=max(atol, 1e-4), rtol=max(rtol, 1e-4))


def test_pfaffian_direct_matches_recursive(skew_matrix, tol):
    """``pfaffian_direct`` (and ``pfaffian(method='direct')``) match the recursive
    reference for the closed-form sizes N in {2, 4}.
    """
    N = skew_matrix.shape[0]
    if N not in (2, 4):
        pytest.skip("pfaffian_direct only supports N in {2, 4}")

    atol, rtol = tol
    pf_direct = pfaffian_direct(skew_matrix)
    pf_dispatch = pfaffian(skew_matrix, method="direct")
    pf_rec = pfaffian_recursive(skew_matrix)

    assert torch.allclose(pf_direct, pf_rec, atol=atol, rtol=rtol)
    assert torch.allclose(pf_dispatch, pf_rec, atol=atol, rtol=rtol)


def test_pfaffian_dispatcher_methods_agree(skew_matrix, tol):
    """``pfaffian(M, method=...)`` returns the same value across all routes.

    ``method='auto'`` selects ``direct`` for N<=4 and ``LTL`` otherwise.
    ``method='recursive'`` is gated on N to keep the O(N!) cost reasonable.
    """
    N = skew_matrix.shape[0]
    atol, rtol = tol

    pf_ltl = pfaffian(skew_matrix, method="LTL")
    pf_auto = pfaffian(skew_matrix, method="auto")
    assert torch.allclose(pf_auto, pf_ltl, atol=atol, rtol=rtol)

    if N <= 4:
        pf_direct = pfaffian(skew_matrix, method="direct")
        assert torch.allclose(pf_direct, pf_ltl, atol=atol, rtol=rtol)

    if N <= 10:
        pf_rec = pfaffian(skew_matrix, method="recursive")
        assert torch.allclose(pf_rec, pf_ltl, atol=atol, rtol=rtol)


def test_pfaffian_dispatcher_rejects_unknown_method(skew_matrix):
    """Dispatcher raises on unknown method names."""
    with pytest.raises(ValueError):
        pfaffian(skew_matrix, method="not-a-method")


def test_log_pfaffian_odd_dim(seed, dtype):
    """``log_pfaffian`` short-circuits to (sign=1, logpf=-inf) for odd N.

    Exercises the odd-N branch of ``batched_log_pfaffian_ltl`` (and, when
    routed through it on CUDA, of ``triton_log_pfaffian``).
    """
    for N in (3, 5):
        torch.manual_seed(seed)
        A = torch.randn(N, N, dtype=dtype)
        A = A - A.T

        sign, logpf = log_pfaffian(A, backend="pytorch")
        assert sign.item() == 1.0, f"N={N}: sign={sign.item()}, expected 1.0"
        assert torch.isinf(logpf) and logpf.item() < 0, (
            f"N={N}: logpf={logpf.item()}, expected -inf"
        )

        # Same for a batch
        A_batch = A.unsqueeze(0).expand(3, -1, -1).contiguous()
        sign_b, logpf_b = log_pfaffian(A_batch, backend="pytorch")
        assert sign_b.shape == (3,)
        assert (sign_b == 1.0).all()
        assert torch.isinf(logpf_b).all() and (logpf_b < 0).all()
