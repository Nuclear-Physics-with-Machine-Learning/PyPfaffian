import torch

from . utils import batched_log_pfaffian_ltl, pfaffian_LTL


# Check if Triton is available
_triton_available = False
try:
    import triton
    _triton_available = True
except ImportError:
    pass


class LogPfaffian(torch.autograd.Function):
    """Differentiable log-Pfaffian via Parlett-Reid LTL decomposition.

    Composes with torch.vmap, torch.compile, and torch.func.grad/jacrev.
    Uses the new-style forward/setup_context API to avoid graph breaks.

    Forward: Parlett-Reid LTL with partial pivoting (batched).
    Backward: Analytical gradient d(log|pf(A)|)/dA_ij = 1/2 * (A^{-1})_ij
    """

    @staticmethod
    def forward(A):
        """Compute (sign, log|pf(A)|) for input A.

        Args:
            A: skew-symmetric matrix, shape (N, N) or (batch, N, N)

        Returns:
            sign: shape () or (batch,)
            logpf: shape () or (batch,)
        """
        sign, logpf = batched_log_pfaffian_ltl(A)
        return sign, logpf

    @staticmethod
    def setup_context(ctx, inputs, output):
        A, = inputs
        sign, logpf = output
        ctx.save_for_backward(A)

    @staticmethod
    def backward(ctx, grad_sign, grad_logpf):
        """Analytical backward via the Jacobi-like formula for skew-symmetric A:
        d(log|pf(A)|)/dA_ij = -1/2 * (A^{-1})_ij
        """
        A, = ctx.saved_tensors
        A_inv = torch.linalg.inv(A)
        # grad_logpf may be scalar or (batch,); broadcast to (..., 1, 1)
        if grad_logpf.dim() == 0:
            return -0.5 * A_inv * grad_logpf
        else:
            return -0.5 * A_inv * grad_logpf.unsqueeze(-1).unsqueeze(-1)

    @staticmethod
    def vmap(info, in_dims, A):
        """vmap batching rule: forward already handles batched input."""
        bdim = in_dims[0]
        if bdim is not None and bdim != 0:
            A = A.movedim(bdim, 0)
        elif bdim is None:
            # Unbatched input inside vmap: add batch dim
            A = A.unsqueeze(0)

        # Prefer Triton inside vmap: it runs entirely on-GPU and avoids
        # the overhead of the batched PyTorch LTL path.  However, inductor
        # can't lower a Triton kernel inside vmap, so fall back to the
        # (now tensorized) PyTorch path under torch.compile.
        if A.is_cuda and _triton_available and not torch.compiler.is_compiling():
            from .triton_pfaffian import triton_log_pfaffian
            sign, logpf = triton_log_pfaffian(A)
        else:
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
        from . utils import pfaffian_direct
        return pfaffian_direct(M)
    elif method == "recursive":
        from . utils import pfaffian_recursive
        return pfaffian_recursive(M)
    else:
        raise ValueError(f"Method {method} not supported")


def log_pfaffian(M: torch.Tensor, backend: str = "auto"):
    """Compute (sign, log|pf(M)|) for a skew-symmetric matrix M.

    Uses the Parlett-Reid LTL algorithm. Supports vmap, torch.compile,
    and autograd differentiation.

    Args:
        M: skew-symmetric matrix, shape (N, N) or (batch, N, N)
        backend: "auto" (default), "pytorch", or "triton"
            - "auto": use Triton on CUDA if available, else pure PyTorch
            - "pytorch": always use pure PyTorch backend
            - "triton": use Triton backend (requires GPU + triton)

    Returns:
        (sign, log|pf|) tuple
    """
    if backend == "auto":
        if M.is_cuda and _triton_available and not torch.compiler.is_compiling():
            # Triton needs raw CUDA pointers; functorch transforms (vmap,
            # grad, …) wrap tensors in BatchedTensor / FunctionalTensor
            # that lack storage.  Fall back to PyTorch in that case.
            try:
                M.data_ptr()
                backend = "triton"
            except RuntimeError:
                backend = "pytorch"
        else:
            backend = "pytorch"

    if backend == "triton":
        from . triton_pfaffian import triton_log_pfaffian
        return triton_log_pfaffian(M)
    else:
        return LogPfaffian.apply(M)
