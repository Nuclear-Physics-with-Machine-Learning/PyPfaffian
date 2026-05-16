"""torch.compile + vmap composition tests for the log-Pfaffian.

These tests guard the two fixes that landed together:

1. Fix A: the Triton kernel is wrapped as a `torch.library.custom_op` so it
   composes with `torch.compile(vmap(...))` on CUDA without being silently
   bypassed for the PyTorch path.
2. Fix B: the PyTorch fallback in `batched_log_pfaffian_ltl` no longer uses
   `aten::scatter_.src` (which has no functorch batching rule), so under
   vmap+compile we do not fall back to a per-walker Python loop.

CPU tests exercise Fix B; CUDA tests exercise Fix A.
"""

import warnings

import pytest
import torch
from torch.func import grad

from torchpfaff import log_pfaffian
from torchpfaff.pfaffian import LogPfaffian, _triton_available


_UPSTREAM_XFAIL_REASON = (
    "Upstream PyTorch bug: torch.compile + vmap + grad of an autograd.Function "
    "with a custom backward bypasses LogPfaffian.backward.  Dynamo inlines the "
    "forward and auto-diffs through torch.argmax (pivot selection), yielding "
    "mostly-zero gradients.  register_autograd is also broken with functorch in "
    "this PT version (the auto-generated Function lacks setup_context).  The "
    "user-facing path `log_pfaffian()` is workaround'd via @torch._dynamo.disable; "
    "see the via_log_pfaffian tests below.  Remove these xfails once upstream "
    "PyTorch is fixed."
)


@pytest.fixture
def compile_batch(seed):
    """Fixed-size batch for compile tests.

    The default `skew_matrix_batch` fixture parametrizes over six values of
    N (2, 4, 6, 12, 16, 20).  torch.compile recompiles per input shape, so
    using the parametrized fixture would balloon test time without adding
    coverage.  Pick a single moderate N and reuse it everywhere.
    """
    torch.manual_seed(seed)
    M = torch.randn(5, 8, 8, dtype=torch.float64)
    return M - M.transpose(-1, -2)


@pytest.fixture(autouse=True)
def _reset_dynamo():
    """Clear Dynamo caches between tests so each test re-traces from scratch."""
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _logpf_fn(A):
    """Return only the log|pf| component, for use with torch.func.grad.

    Calls `LogPfaffian.apply` directly -- exercises the path that triggers
    the two upstream PyTorch bugs documented at the top of this file.
    """
    _sign, logpf = LogPfaffian.apply(A)
    return logpf


def _logpf_via_top_level(A):
    """Same scalar output as `_logpf_fn` but routed through the workaround'd
    user-facing `log_pfaffian()` helper.  Under torch.compile this triggers a
    graph break (because `log_pfaffian` is @torch._dynamo.disable'd) so eager
    autograd correctly uses `LogPfaffian.backward`.
    """
    _sign, logpf = log_pfaffian(A)
    return logpf


# ----------------------------------------------------------------------
# CPU: exercises Fix B (out-of-place scatter in batched_log_pfaffian_ltl)
# ----------------------------------------------------------------------


def test_compile_vmap_forward_cpu_matches_eager(compile_batch):
    """torch.compile(vmap(LogPfaffian.apply)) matches the eager vmap result on CPU."""
    ref_sign, ref_logpf = torch.vmap(LogPfaffian.apply)(compile_batch)

    compiled = torch.compile(torch.vmap(LogPfaffian.apply), fullgraph=False)
    sign, logpf = compiled(compile_batch)

    torch.testing.assert_close(sign, ref_sign)
    torch.testing.assert_close(logpf, ref_logpf, atol=1e-10, rtol=1e-10)


def test_compile_vmap_emits_no_scatter_batching_warning(compile_batch):
    """Regression guard: no aten::scatter_.src batching-fallback warning."""
    compiled = torch.compile(torch.vmap(LogPfaffian.apply), fullgraph=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compiled(compile_batch)

    offenders = [w for w in caught if "scatter_.src" in str(w.message)]
    assert not offenders, (
        f"Unexpected scatter_.src batching warnings: "
        f"{[str(w.message) for w in offenders]}"
    )


def test_vmap_grad_log_pfaffian_eager_cpu(compile_batch):
    """Eager vmap(grad(log_pfaffian)) matches the analytical -1/2 A^{-1}."""
    grads = torch.vmap(grad(_logpf_fn))(compile_batch)
    expected = -0.5 * torch.linalg.inv(compile_batch)
    torch.testing.assert_close(grads, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.xfail(reason=_UPSTREAM_XFAIL_REASON, strict=False)
def test_compile_vmap_grad_matches_eager_cpu(compile_batch):
    """The torch_qmc usage pattern: torch.compile(vmap(grad(...))).

    Marked xfail: hits the upstream bug because it calls `LogPfaffian.apply`
    directly inside `_logpf_fn`.  See `test_compile_vmap_grad_via_log_pfaffian_cpu`
    for the workaround'd version that is expected to pass.
    """
    eager = torch.vmap(grad(_logpf_fn))(compile_batch)
    compiled = torch.compile(torch.vmap(grad(_logpf_fn)), fullgraph=False)(
        compile_batch
    )
    torch.testing.assert_close(compiled, eager, atol=1e-8, rtol=1e-8)


def test_compile_vmap_grad_via_log_pfaffian_cpu(compile_batch):
    """compile(vmap(grad(log_pfaffian))) on CPU.

    Goes through the @torch._dynamo.disable'd `log_pfaffian()` helper, so the
    Pfaffian call becomes a graph break under compile and eager autograd
    correctly uses `LogPfaffian.backward`.  This is the safe API path that
    `torch_qmc` actually uses (`Pfaffian.__call__` -> `log_pfaffian(matrix)`).
    """
    eager = torch.vmap(grad(_logpf_via_top_level))(compile_batch)
    compiled = torch.compile(
        torch.vmap(grad(_logpf_via_top_level)), fullgraph=False
    )(compile_batch)
    torch.testing.assert_close(compiled, eager, atol=1e-8, rtol=1e-8)


# ----------------------------------------------------------------------
# CPU: standalone compile(forward) and compile(grad) WITHOUT vmap
# ----------------------------------------------------------------------


def test_compile_forward_cpu_matches_eager(compile_batch):
    """``torch.compile(LogPfaffian.apply)`` (no vmap) matches eager.

    Covers the plain compiled-forward path on a batched input, distinct
    from the compile(vmap(forward)) tests above.
    """
    ref_sign, ref_logpf = LogPfaffian.apply(compile_batch)

    compiled = torch.compile(LogPfaffian.apply, fullgraph=False)
    sign, logpf = compiled(compile_batch)

    torch.testing.assert_close(sign, ref_sign)
    torch.testing.assert_close(logpf, ref_logpf, atol=1e-10, rtol=1e-10)


def test_compile_grad_via_log_pfaffian_cpu(compile_batch):
    """``compile(grad(log_pfaffian))`` (no vmap) matches eager grad.

    Routes through the @torch._dynamo.disable'd ``log_pfaffian()`` helper
    so the Pfaffian itself becomes a graph break and eager autograd uses
    ``LogPfaffian.backward``.  Pins the non-vmap'd compiled-grad path.
    """
    A_single = compile_batch[0]
    eager = grad(_logpf_via_top_level)(A_single)
    compiled = torch.compile(grad(_logpf_via_top_level), fullgraph=False)(A_single)
    torch.testing.assert_close(compiled, eager, atol=1e-8, rtol=1e-8)


# ----------------------------------------------------------------------
# CUDA: exercises Fix A (Triton custom_op composes with torch.compile)
# ----------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_custom_op_direct_cuda(compile_batch):
    """The custom op dispatches to Triton on CUDA and matches LogPfaffian.apply.

    LogPfaffian.apply now forwards into the same custom op, so this test
    primarily exercises the Triton dispatch path and the register_autograd
    pass-through against an independent eager call.
    """
    if not _triton_available:
        pytest.skip("Triton not installed")

    from torchpfaff.pfaffian import _log_pfaffian_op

    A_cuda = compile_batch.cuda()
    sign_t, logpf_t = _log_pfaffian_op(A_cuda)
    sign_p, logpf_p = LogPfaffian.apply(A_cuda)

    torch.testing.assert_close(sign_t, sign_p)
    torch.testing.assert_close(logpf_t, logpf_p, atol=1e-8, rtol=1e-8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compile_vmap_uses_triton_cuda(compile_batch):
    """Compile + vmap on CUDA matches eager vmap (which routes to Triton)."""
    if not _triton_available:
        pytest.skip("Triton not installed")

    A_cuda = compile_batch.cuda()
    ref_sign, ref_logpf = torch.vmap(LogPfaffian.apply)(A_cuda)

    compiled = torch.compile(torch.vmap(LogPfaffian.apply), fullgraph=False)
    sign, logpf = compiled(A_cuda)

    torch.testing.assert_close(sign, ref_sign)
    torch.testing.assert_close(logpf, ref_logpf, atol=1e-8, rtol=1e-8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.xfail(reason=_UPSTREAM_XFAIL_REASON, strict=False)
def test_compile_vmap_grad_cuda(compile_batch):
    """Full torch_qmc-shape composition on CUDA: compile(vmap(grad(log_pfaffian))).

    Marked xfail for the same upstream reason as the CPU variant.  See
    `test_compile_vmap_grad_via_log_pfaffian_cuda` for the workaround'd version.
    """
    if not _triton_available:
        pytest.skip("Triton not installed")

    A_cuda = compile_batch.cuda()
    eager = torch.vmap(grad(_logpf_fn))(A_cuda)
    compiled = torch.compile(torch.vmap(grad(_logpf_fn)), fullgraph=False)(A_cuda)
    torch.testing.assert_close(compiled, eager, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compile_vmap_grad_via_log_pfaffian_cuda(compile_batch):
    """compile(vmap(grad(log_pfaffian))) on CUDA via the safe top-level helper.

    Same workaround pattern as the CPU variant; this test confirms the
    @torch._dynamo.disable graph break also works on CUDA, where the underlying
    forward dispatches to Triton.
    """
    if not _triton_available:
        pytest.skip("Triton not installed")

    A_cuda = compile_batch.cuda()
    eager = torch.vmap(grad(_logpf_via_top_level))(A_cuda)
    compiled = torch.compile(
        torch.vmap(grad(_logpf_via_top_level)), fullgraph=False
    )(A_cuda)
    torch.testing.assert_close(compiled, eager, atol=1e-6, rtol=1e-6)
