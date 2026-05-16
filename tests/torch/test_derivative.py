import pytest
import torch

from torchpfaff.pfaffian import LogPfaffian
from torchpfaff.utils import pfaffian_LTL

_FWD_AD_XFAIL = (
    "LogPfaffian does not register a forward-mode AD rule (no jvp staticmethod), "
    "and the underlying _log_pfaffian_op is intentionally not autograd-registered.  "
    "torch.func.jvp / jacfwd therefore have no JVP path to dispatch through.  "
    "Documented gap; remove this xfail when forward-mode AD is added."
)


def test_log_pfaffian_gradcheck(N, seed, dtype):
    """Verify LogPfaffian backward against numerical finite differences.

    Parametrize through antisymmetrization so gradcheck's per-element
    perturbations never break skew-symmetry.  ``gradcheck`` uses a tiny
    epsilon (~1e-6 default) which is below the float32 noise floor, so
    we restrict this test to float64.
    """
    if dtype != torch.float64:
        pytest.skip("gradcheck epsilon is below the float32 noise floor")

    torch.manual_seed(seed)
    M = torch.randn(N, N, dtype=dtype, requires_grad=True)

    def logpf_via_skew(M):
        A = M - M.T
        _sign, logpf = LogPfaffian.apply(A)
        return logpf

    assert torch.autograd.gradcheck(logpf_via_skew, (M,), atol=1e-5, rtol=1e-4)


def test_log_pfaffian_vjp_analytical(skew_matrix, tol):
    """Gradient of log|pf(A)| w.r.t. A should equal -1/2 * A^{-1}."""
    atol, rtol = tol
    A = skew_matrix.clone().requires_grad_(True)

    _sign, logpf = LogPfaffian.apply(A)
    logpf.backward()

    expected = -0.5 * torch.linalg.inv(skew_matrix)
    assert torch.allclose(A.grad, expected, atol=atol, rtol=rtol), (
        f"max error: {(A.grad - expected).abs().max().item()}"
    )


def test_pfaffian_vjp_contract_numerical(N, seed, dtype):
    """Contract the analytical VJP with a tangent and compare to a central-FD JVP.

    Despite the historical name, this test computes a VJP via ``.backward()``
    and contracts it with a tangent direction, which equals the JVP only
    because the output is scalar.  Renamed for accuracy; a true forward-mode
    JVP test lives in ``test_log_pfaffian_jvp_via_torch_func`` below.
    """
    if N < 4:
        pytest.skip("Tangent perturbs entry [0,3]; need N >= 4")

    torch.manual_seed(int(seed))
    A = torch.randn(N, N, dtype=dtype)
    A = A - A.T

    # Skew-symmetric tangent touching [0,3] and [3,0]
    tangent = torch.zeros_like(A)
    tangent[0, 3] = 1.0
    tangent = tangent - tangent.T

    # Analytical VJP-contracted-with-tangent via backward
    A_diff = A.clone().requires_grad_(True)
    sign, logpf = LogPfaffian.apply(A_diff)
    pf = sign * torch.exp(logpf)
    pf.backward()
    analytical_jvp = (A_diff.grad * tangent).sum()

    # Central finite-difference, dtype-aware kick (~ sqrt(eps))
    kick = 1e-3 if dtype == torch.float32 else 1e-5
    pf_plus = pfaffian_LTL(A + 0.5 * kick * tangent)
    pf_minus = pfaffian_LTL(A - 0.5 * kick * tangent)
    numerical_jvp = (pf_plus - pf_minus) / kick

    rel = max(numerical_jvp.abs().item(), 1.0)
    fd_atol = (1e-2 if dtype == torch.float32 else 1e-4) * rel
    assert torch.allclose(analytical_jvp, numerical_jvp, atol=fd_atol), (
        f"analytical={analytical_jvp.item()}, numerical={numerical_jvp.item()}"
    )


def test_log_pfaffian_vmap(skew_matrix_batch, tol):
    """vmap over LogPfaffian.apply matches per-matrix calls."""
    atol, rtol = tol
    batch = skew_matrix_batch.shape[0]

    sign_vmap, logpf_vmap = torch.vmap(LogPfaffian.apply)(skew_matrix_batch)

    for i in range(batch):
        sign_i, logpf_i = LogPfaffian.apply(skew_matrix_batch[i])
        assert torch.allclose(sign_vmap[i], sign_i), (
            f"batch[{i}] sign: vmap={sign_vmap[i].item()}, single={sign_i.item()}"
        )
        assert torch.allclose(logpf_vmap[i], logpf_i, atol=atol, rtol=rtol), (
            f"batch[{i}] logpf: vmap={logpf_vmap[i].item()}, single={logpf_i.item()}"
        )


# ----------------------------------------------------------------------
# torch.func: jacrev / jacfwd / jvp
# ----------------------------------------------------------------------


def _logpf_scalar(A):
    """Return the scalar log|pf(A)| for use with jacrev/jacfwd/jvp."""
    _sign, logpf = LogPfaffian.apply(A)
    return logpf


def test_log_pfaffian_jacrev_matches_analytical(skew_matrix, tol):
    """``torch.func.jacrev`` of log|pf(A)| equals the analytical -1/2 * A^{-1}.

    Pins the reverse-mode functional API path (which composes with vmap),
    independent of the imperative ``.backward()`` path.
    """
    atol, rtol = tol
    jac = torch.func.jacrev(_logpf_scalar)(skew_matrix)
    expected = -0.5 * torch.linalg.inv(skew_matrix)
    assert torch.allclose(jac, expected, atol=atol, rtol=rtol), (
        f"max error: {(jac - expected).abs().max().item()}"
    )


def test_log_pfaffian_jacrev_via_vmap(skew_matrix_batch, tol):
    """``vmap(jacrev(...))`` matches per-matrix jacrev.

    The torch_qmc usage shape relies on this composition for per-walker
    Hessians and gradients.
    """
    atol, rtol = tol
    batch_jac = torch.vmap(torch.func.jacrev(_logpf_scalar))(skew_matrix_batch)

    for i in range(skew_matrix_batch.shape[0]):
        ref = torch.func.jacrev(_logpf_scalar)(skew_matrix_batch[i])
        assert torch.allclose(batch_jac[i], ref, atol=atol, rtol=rtol)


@pytest.mark.xfail(reason=_FWD_AD_XFAIL, strict=False)
def test_log_pfaffian_jacfwd_matches_jacrev(skew_matrix, tol):
    """Forward-mode jacobian ``jacfwd`` equals reverse-mode ``jacrev``.

    Currently expected to xfail because LogPfaffian has no ``jvp`` rule.
    """
    atol, rtol = tol
    jac_fwd = torch.func.jacfwd(_logpf_scalar)(skew_matrix)
    jac_rev = torch.func.jacrev(_logpf_scalar)(skew_matrix)
    assert torch.allclose(jac_fwd, jac_rev, atol=atol, rtol=rtol)


@pytest.mark.xfail(reason=_FWD_AD_XFAIL, strict=False)
def test_log_pfaffian_jvp_via_torch_func(N, seed, dtype):
    """True forward-mode JVP via ``torch.func.jvp``.

    The directional derivative of log|pf(A)| in tangent direction V is
    ``sum_ij (-1/2)(A^{-1})_ij * V_ij``.  Currently xfails because no
    forward-mode AD rule is registered; kept as a regression guard.
    """
    if N < 4:
        pytest.skip("Need N >= 4 for the chosen tangent")

    torch.manual_seed(int(seed))
    A = torch.randn(N, N, dtype=dtype)
    A = A - A.T

    tangent = torch.zeros_like(A)
    tangent[0, 3] = 1.0
    tangent = tangent - tangent.T

    (sign, logpf), (sign_dot, logpf_dot) = torch.func.jvp(
        LogPfaffian.apply, (A,), (tangent,)
    )

    expected_dot = (-0.5 * torch.linalg.inv(A) * tangent).sum()
    fd_atol = 1e-3 if dtype == torch.float32 else 1e-6
    assert torch.allclose(logpf_dot, expected_dot, atol=fd_atol)


# ----------------------------------------------------------------------
# Double backward
# ----------------------------------------------------------------------


def test_log_pfaffian_double_backward(skew_matrix, tol):
    """Second derivative of log|pf(A)| executes and matches the analytical
    formula ``d^2(log|pf|)/dA^2 . V = (1/2) * A^{-1} V A^{-1}``.

    Exercises the create_graph path through ``LogPfaffian.backward`` ->
    ``torch.linalg.inv`` -> autograd.
    """
    N = skew_matrix.shape[0]
    if N < 4:
        pytest.skip("Need N >= 4 for a non-trivial tangent")

    atol, rtol = tol

    A = skew_matrix.clone().requires_grad_(True)
    _sign, logpf = LogPfaffian.apply(A)

    (g,) = torch.autograd.grad(logpf, A, create_graph=True)

    # Skew-symmetric tangent V touching [0, N-1]
    V = torch.zeros_like(skew_matrix)
    V[0, N - 1] = 1.0
    V = V - V.T

    scalar = (g * V).sum()
    (g2,) = torch.autograd.grad(scalar, A)

    A_inv = torch.linalg.inv(skew_matrix)
    expected = 0.5 * (A_inv @ V @ A_inv)

    # Loosen for float32 (two A^{-1} contractions accumulate noise).
    looser = (max(atol, 1e-3), max(rtol, 1e-3))
    assert torch.allclose(g2, expected, atol=looser[0], rtol=looser[1]), (
        f"max error: {(g2 - expected).abs().max().item()}"
    )


# ----------------------------------------------------------------------
# vmap rule: unbatched-input branch
# ----------------------------------------------------------------------


def test_log_pfaffian_vmap_unbatched_branch(skew_matrix, tol):
    """Exercise ``LogPfaffian.vmap`` with an unbatched input (in_dim=None).

    When ``LogPfaffian.apply`` is invoked under ``torch.vmap`` but its
    matrix argument does NOT depend on the vmapped variable, in_dims[0]
    is None and the rule must broadcast: it unsqueezes a batch dim,
    runs the kernel, then squeezes the output back.

    Without this branch the call would error or silently return the
    wrong shape.
    """
    atol, rtol = tol
    A_const = skew_matrix
    sign_ref, logpf_ref = LogPfaffian.apply(A_const)

    batch_size = 4
    dummy = torch.zeros(batch_size, dtype=skew_matrix.dtype)

    def fn(_dummy_scalar):
        return LogPfaffian.apply(A_const)

    sign_b, logpf_b = torch.vmap(fn)(dummy)

    assert sign_b.shape == (batch_size,)
    assert logpf_b.shape == (batch_size,)
    assert torch.allclose(sign_b, sign_ref.expand(batch_size))
    assert torch.allclose(
        logpf_b, logpf_ref.expand(batch_size), atol=atol, rtol=rtol
    )
