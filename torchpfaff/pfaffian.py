# Check if Triton is available (use find_spec to avoid an unused-import lint)
import importlib.util

import torch

from .utils import batched_log_pfaffian_ltl, pfaffian_LTL

_triton_available = importlib.util.find_spec("triton") is not None


@torch.library.custom_op(
    "torchpfaff::log_pfaffian",
    mutates_args=(),
)
def _log_pfaffian_op(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-Pfaffian as a torch.library.custom_op (forward only).

    Dispatches to Triton on CUDA (when triton is installed) and to the
    pure-PyTorch batched LTL implementation otherwise.  This op is
    intentionally NOT autograd-registered (see the NOTE below); all
    differentiation is provided by the outer ``LogPfaffian`` autograd
    Function, which calls into this op from ``forward`` and from its
    explicit vmap rule.  Registering this op gives torch.compile a single
    opaque node with shape inference via ``register_fake``.
    """
    if A.is_cuda and _triton_available:
        from .triton_pfaffian import triton_log_pfaffian

        return triton_log_pfaffian(A)
    return batched_log_pfaffian_ltl(A)


@_log_pfaffian_op.register_fake
def _(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Meta/shape-inference implementation used during Dynamo tracing.

    Handles both unbatched (N, N) and batched (B, N, N) inputs by taking
    everything but the trailing two dims as the batch shape.
    """
    batch_shape = A.shape[:-2]
    sign = torch.empty(batch_shape, dtype=A.dtype, device=A.device)
    logpf = torch.empty(batch_shape, dtype=A.dtype, device=A.device)
    return sign, logpf


# NOTE: We intentionally do NOT call _log_pfaffian_op.register_autograd().
# In this PyTorch version, register_autograd's auto-generated
# torch.autograd.Function (torch/_library/autograd.py:make_autograd_impl)
# is constructed without a `setup_context` staticmethod, so under any
# functorch transform (vmap, grad, jvp, ...) torch.autograd.Function.apply
# raises "must override the setup_context staticmethod".
# Autograd through this op is handled by the outer `LogPfaffian` autograd
# Function below; under torch.compile we additionally graph-break at
# `log_pfaffian()` (see the @torch._dynamo.disable on that helper).


class LogPfaffian(torch.autograd.Function):
    """Differentiable log-Pfaffian via Parlett-Reid LTL decomposition.

    Composes with torch.vmap, torch.compile, and torch.func.grad/jacrev.
    Uses the new-style forward/setup_context API to avoid graph breaks.

    Forward: Parlett-Reid LTL with partial pivoting (batched).
    Backward: Analytical gradient d(log|pf(A)|)/dA_ij = -1/2 * (A^{-1})_ij
    JVP: Analytical contraction of that gradient with the matrix tangent.
    """

    @staticmethod
    def forward(A):
        """Compute (sign, log|pf(A)|) for input A.

        Delegates to `_log_pfaffian_op` which dispatches to Triton on CUDA
        and to the pure-PyTorch batched LTL implementation otherwise.

        Args:
            A: skew-symmetric matrix, shape (N, N) or (batch, N, N)

        Returns:
            sign: shape () or (batch,)
            logpf: shape () or (batch,)
        """
        return _log_pfaffian_op(A)

    @staticmethod
    def setup_context(ctx, inputs, output):
        (A,) = inputs
        ctx.save_for_backward(A)
        ctx.save_for_forward(A)

    @staticmethod
    def backward(ctx, grad_sign, grad_logpf):
        """Analytical backward via the Jacobi-like formula for skew-symmetric A:
        d(log|pf(A)|)/dA_ij = -1/2 * (A^{-1})_ij

        Branchless: reshape grad_logpf to add two trailing length-1 dims so
        it broadcasts over the (N, N) tail of A_inv for any rank of
        grad_logpf (0-dim, 1-dim, ...).  ``grad_sign`` is unused because
        the sign is piecewise constant (non-differentiable).
        """
        (A,) = ctx.saved_tensors
        A_inv = torch.linalg.inv(A)
        grad_logpf_b = grad_logpf.reshape(*grad_logpf.shape, 1, 1)
        return -0.5 * A_inv * grad_logpf_b

    @staticmethod
    def jvp(ctx, grad_A):
        """Forward-mode derivative of ``(sign(A), log|pf(A)|)``.

        The Pfaffian sign is piecewise constant.  For the log magnitude,
        ``d log|pf(A)| = -1/2 sum_ij (A^-1)_ij dA_ij`` for a
        skew-symmetric matrix and tangent.
        """
        (A,) = ctx.saved_tensors
        grad_logpf = (-0.5 * torch.linalg.inv(A) * grad_A).sum(dim=(-2, -1))
        return torch.zeros_like(grad_logpf), grad_logpf

    @staticmethod
    def vmap(info, in_dims, A):
        """Batch the matrix input while preserving the outer autograd Function.

        Re-entering ``LogPfaffian.apply`` on the unwrapped physical batch makes
        both ``vmap(grad(...))`` and ``grad(vmap(...))`` use the analytical
        backward.  Calling the raw custom op here would support forward-only
        vmap but lose autograd under an outer functional transform.

        ``register_fake`` on the underlying op lets Dynamo perform shape
        inference under ``torch.compile`` without executing the Triton
        kernel.
        """
        bdim = in_dims[0]
        if bdim is not None and bdim != 0:
            A = A.movedim(bdim, 0)
        elif bdim is None:
            # Unbatched input inside vmap: add batch dim
            A = A.unsqueeze(0)

        # Re-enter the outer autograd Function on the physical (unwrapped)
        # batch tensor.  Calling the raw custom op here would make
        # grad(vmap(LogPfaffian)) lose the analytical backward.
        sign, logpf = LogPfaffian.apply(A)

        if bdim is None:
            return (sign.squeeze(0), logpf.squeeze(0)), (None, None)
        return (sign, logpf), (0, 0)


def pfaffian(M: torch.Tensor, method: str = "LTL"):
    """Compute and return the Pfaffian of a skew-symmetric matrix M.

    Args:
        M: skew-symmetric matrix, shape (N, N)
        method: "LTL" (default), "direct" (N<=4), or "recursive" (reference)

    Returns:
        Scalar Pfaffian value
    """
    if method == "auto":
        N = M.shape[-1]
        if N <= 4:
            method = "direct"
        else:
            method = "LTL"

    if method == "LTL":
        return pfaffian_LTL(M)
    elif method == "direct":
        from .utils import pfaffian_direct

        return pfaffian_direct(M)
    elif method == "recursive":
        from .utils import pfaffian_recursive

        return pfaffian_recursive(M)
    else:
        raise ValueError(f"Method {method} not supported")


@torch._dynamo.disable
def log_pfaffian(M: torch.Tensor, backend: str = "auto"):
    """Compute (sign, log|pf(M)|) for a skew-symmetric matrix M.

    Uses the Parlett-Reid LTL algorithm.  Supports vmap, torch.compile,
    and autograd differentiation.

    Args:
        M: skew-symmetric matrix, shape (N, N) or (batch, N, N)
        backend: "auto" (default), "pytorch", or "triton"
            - "auto": go through LogPfaffian.apply, which routes to the
              custom op (Triton on CUDA, pure PyTorch otherwise) and
              picks up the vmap rule for BatchedTensor inputs.
            - "pytorch": same as "auto"; kept for backward compat.
            - "triton": force the Triton kernel (requires GPU + triton).

    Returns:
        (sign, log|pf|) tuple

    Notes:
        This function is intentionally marked ``@torch._dynamo.disable``
        so it survives ``torch.compile(vmap(grad(...)))`` correctly.  Two
        upstream PyTorch bugs interact in that composition:

        1. ``torch.library.register_autograd``'s auto-generated
           autograd.Function (``torch/_library/autograd.py:make_autograd_impl``)
           lacks a ``setup_context`` staticmethod, so it raises under any
           functorch transform.
        2. Without ``register_autograd``, Dynamo inlines
           ``LogPfaffian.forward`` and differentiates through
           ``torch.argmax`` (the pivot selection inside
           ``batched_log_pfaffian_ltl``) instead of using the analytical
           custom backward, yielding mostly-zero gradients silently.

        Forcing a graph break at this call boundary lets eager autograd
        run ``LogPfaffian.apply`` correctly.  The surrounding code (matrix
        construction, MLPs, ...) stays compiled; only this one call is
        eager.  Remove the disable once either upstream bug is fixed.
    """
    if backend == "auto" or backend == "pytorch":
        return LogPfaffian.apply(M)
    if backend == "triton":
        from .triton_pfaffian import triton_log_pfaffian

        return triton_log_pfaffian(M)
    raise ValueError(f"Unknown backend: {backend!r}")
