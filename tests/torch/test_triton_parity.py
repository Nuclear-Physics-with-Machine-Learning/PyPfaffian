"""Triton kernel vs. pure-PyTorch reference parity tests.

The compile-side CUDA tests in ``test_compile.py`` only compare Triton to
``LogPfaffian.apply`` on CUDA, which itself dispatches into Triton -- so
that suite cannot catch a divergence between the two backends.  The tests
here pin the Triton kernel against the pure-PyTorch ``batched_log_pfaffian_ltl``
on CPU (the reference implementation), which is independently checked by
``test_torch_correctness.py``.

All tests are CUDA-only and gated on ``triton`` being installed.
"""

import pytest
import torch

from torchpfaff.pfaffian import LogPfaffian, _triton_available
from torchpfaff.utils import batched_log_pfaffian_ltl

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(not _triton_available, reason="Triton not installed"),
]


def _tol_for(dtype):
    """Triton-vs-PyTorch parity tolerances; loosened vs. eager-only checks
    to absorb FMA reordering and any kernel-internal rounding differences.
    """
    if dtype == torch.float32:
        return 1e-3, 1e-3
    return 1e-7, 1e-7


def test_triton_vs_pytorch_forward(N, seed, dtype):
    """The Triton kernel and the pure-PyTorch LTL reference agree on (sign, logpf).

    Generates a skew matrix on CPU, evaluates the reference there, then
    evaluates Triton on CUDA, and compares.
    """
    from torchpfaff.triton_pfaffian import triton_log_pfaffian

    torch.manual_seed(seed)
    M = torch.randn(N, N, dtype=dtype)
    A_cpu = M - M.T
    A_cuda = A_cpu.cuda()

    sign_ref, logpf_ref = batched_log_pfaffian_ltl(A_cpu)
    sign_t, logpf_t = triton_log_pfaffian(A_cuda)

    atol, rtol = _tol_for(dtype)
    torch.testing.assert_close(sign_t.cpu(), sign_ref)
    torch.testing.assert_close(logpf_t.cpu(), logpf_ref, atol=atol, rtol=rtol)


def test_triton_vs_pytorch_forward_batched(N, seed, dtype):
    """Same parity check for a batched input."""
    from torchpfaff.triton_pfaffian import triton_log_pfaffian

    torch.manual_seed(seed)
    batch = 5
    M = torch.randn(batch, N, N, dtype=dtype)
    A_cpu = M - M.transpose(-1, -2)
    A_cuda = A_cpu.cuda()

    sign_ref, logpf_ref = batched_log_pfaffian_ltl(A_cpu)
    sign_t, logpf_t = triton_log_pfaffian(A_cuda)

    atol, rtol = _tol_for(dtype)
    torch.testing.assert_close(sign_t.cpu(), sign_ref)
    torch.testing.assert_close(logpf_t.cpu(), logpf_ref, atol=atol, rtol=rtol)


def test_triton_vs_pytorch_grad(N, seed, dtype):
    """``LogPfaffian.apply(A).backward()`` agrees CUDA-vs-CPU.

    The backward goes through ``torch.linalg.inv(A)`` and so is independent
    of the Triton kernel itself; this test pins the saved-tensor path and
    confirms that gradients computed after a Triton forward match those
    computed after a CPU forward.
    """
    torch.manual_seed(seed)
    M = torch.randn(N, N, dtype=dtype)
    A_skew = M - M.T

    A_cpu = A_skew.clone().requires_grad_(True)
    _s, lp_cpu = LogPfaffian.apply(A_cpu)
    lp_cpu.backward()

    A_cuda = A_skew.cuda().clone().requires_grad_(True)
    _s, lp_cuda = LogPfaffian.apply(A_cuda)
    lp_cuda.backward()

    atol, rtol = _tol_for(dtype)
    torch.testing.assert_close(A_cuda.grad.cpu(), A_cpu.grad, atol=atol, rtol=rtol)


def test_triton_odd_dim_short_circuit(seed, dtype):
    """Odd-N short-circuit on the Triton path: sign=1, logpf=-inf."""
    from torchpfaff.triton_pfaffian import triton_log_pfaffian

    for N in (3, 5):
        torch.manual_seed(seed)
        A = torch.randn(N, N, dtype=dtype, device="cuda")
        A = A - A.T
        sign, logpf = triton_log_pfaffian(A)
        assert sign.item() == 1.0
        assert torch.isinf(logpf) and logpf.item() < 0
