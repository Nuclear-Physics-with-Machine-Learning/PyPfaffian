import torch


def pfaffian_direct(A: torch.Tensor):
    """Compute the Pfaffian directly for N<=4.

    Args:
        A: A skew-symmetric matrix of size (N, N)

    Returns:
        Scalar Pfaffian value
    """
    n = A.shape[-1]

    if n % 2 == 1: return torch.zeros((), dtype=A.dtype, device=A.device)
    if n == 2: return -A[..., 1, 0]
    if n == 4:
        return A[...,1,0]*A[...,3,2] - A[...,2,0]*A[...,3,1] + A[...,2,1]*A[...,3,0]
    else:
        raise ValueError("pfaffian_direct only supports N <= 4")


def log_pfaffian_direct(A: torch.Tensor):
    """Compute (sign, log|pf|) directly for N<=4.

    Args:
        A: A skew-symmetric matrix of size (N, N) or (..., N, N)

    Returns:
        (sign, log|pf|) tuple
    """
    n = A.shape[-1]

    if n % 2 == 1:
        return (torch.ones(A.shape[:-2], dtype=A.dtype, device=A.device),
                torch.full(A.shape[:-2], float('-inf'), dtype=A.dtype, device=A.device))

    if n == 2:
        val = -A[..., 1, 0]
        return torch.sign(val), torch.log(torch.abs(val))

    if n == 4:
        val = A[...,1,0]*A[...,3,2] - A[...,2,0]*A[...,3,1] + A[...,2,1]*A[...,3,0]
        return torch.sign(val), torch.log(torch.abs(val))

    raise ValueError("log_pfaffian_direct only supports N <= 4")


def pfaffian_recursive(A: torch.Tensor):
    """Recursive Pfaffian for correctness checking. Not for production use.

    Args:
        A: A skew-symmetric matrix of size (N, N)

    Returns:
        Scalar Pfaffian value
    """
    n = A.shape[0]

    if n == 2:
        return -A[1, 0]

    def delete(_A, index):
        temp_m = torch.cat([_A[:index, :], _A[(index+1):, :]], dim=0)
        return torch.cat([temp_m[:, :index], temp_m[:, (index+1):]], dim=1)

    delete_0 = delete(A, 0)
    this_pf = torch.zeros((), dtype=A.dtype, device=A.device)
    for i in range(1, n):
        a_0j = (-1)**(i) * A[0, i]
        submatrix = delete(delete_0, i - 1)
        this_pf = this_pf + a_0j * pfaffian_recursive(submatrix)

    return -this_pf


def batched_log_pfaffian_ltl(A):
    """Batched Parlett-Reid LTL decomposition for log-Pfaffian.

    Computes (sign, log|pf(A)|) for a batch of skew-symmetric matrices
    using the Parlett-Reid algorithm with partial pivoting.

    This function operates on plain tensors with standard Python control flow.
    It is intended to be called from inside a torch.autograd.Function,
    which encapsulates the data-dependent control flow away from
    functorch transforms.

    Args:
        A: skew-symmetric matrix of shape (N, N) or (batch, N, N)

    Returns:
        sign: shape () or (batch,)
        log_pf: shape () or (batch,)
    """
    # Handle unbatched input
    squeezed = False
    if A.dim() == 2:
        A = A.unsqueeze(0)
        squeezed = True

    batch, n, _ = A.shape

    if n % 2 == 1:
        sign = torch.ones(batch, dtype=A.dtype, device=A.device)
        logpf = torch.full((batch,), float('-inf'), dtype=A.dtype, device=A.device)
        if squeezed:
            return sign.squeeze(0), logpf.squeeze(0)
        return sign, logpf

    # Work on a copy to avoid modifying the input
    W = A.clone()

    pf_sign = torch.ones(batch, dtype=A.dtype, device=A.device)
    log_pf = torch.zeros(batch, dtype=A.dtype, device=A.device)

    arange_n = torch.arange(n, device=A.device)

    for k in range(0, n - 1, 2):
        # --- Pivot selection ---
        # Mask out entries at indices <= k so argmax only sees k+1..n-1
        col = W[:, :, k]  # (batch, n)
        mask = (arange_n <= k).unsqueeze(0)  # (1, n)
        masked_col = torch.where(mask, torch.zeros_like(col), col)
        kp = torch.argmax(torch.abs(masked_col), dim=1)  # (batch,)

        # --- Pivot swap (per matrix in the batch) ---
        # We must iterate over the batch for data-dependent row/col swaps.
        # This is fine: we are inside an autograd.Function boundary,
        # not inside a functorch transform.
        for b in range(batch):
            kp_b = kp[b].item()
            if kp_b != k + 1:
                # Swap rows k+1 and kp_b
                W[b, [k+1, kp_b], :] = W[b, [kp_b, k+1], :]
                # Swap cols k+1 and kp_b
                W[b, :, [k+1, kp_b]] = W[b, :, [kp_b, k+1]]
                pf_sign[b] = -pf_sign[b]

        # --- Gauss vector update (batched) ---
        update = W[:, k, k+1]  # (batch,)

        mu = W[:, k, :] / update.unsqueeze(-1)   # (batch, n) -- row k
        nu = W[:, :, k+1]                         # (batch, n) -- column k+1

        # Rank-2 skew update: A += outer(mu, nu) - outer(nu, mu)
        W = W + torch.bmm(mu.unsqueeze(-1), nu.unsqueeze(-2)) \
              - torch.bmm(nu.unsqueeze(-1), mu.unsqueeze(-2))

        # Accumulate sign and log
        pf_sign = pf_sign * torch.sign(update)
        log_pf = log_pf + torch.log(torch.abs(update))

    if squeezed:
        return pf_sign.squeeze(0), log_pf.squeeze(0)
    return pf_sign, log_pf


# ---- Legacy API wrappers (kept for backward compat with existing callers) ----

def pivot(_A, _k, _kp):
    """Perform a pivot swap on a single matrix. Legacy API."""
    _A = _A.clone()
    temp = _A[_k + 1].clone()
    _A[_k + 1] = _A[_kp]
    _A[_kp] = temp
    temp = _A[:, _k + 1].clone()
    _A[:, _k + 1] = _A[:, _kp]
    _A[:, _kp] = temp
    return _A, -1.0

def form_gauss_vector(_A, _k):
    """Form Gauss vector and return update. Legacy API."""
    pfaffian_update = _A[_k, _k+1]
    mu = _A[_k, :] / pfaffian_update
    nu = _A[:, _k+1]
    _A = _A + torch.outer(mu, nu) - torch.outer(nu, mu)
    return _A, pfaffian_update

def pfaffian_LTL(A):
    """Compute the Pfaffian via Parlett-Reid LTL. Single-matrix, legacy API."""
    n = A.shape[0]
    if n % 2 == 1:
        return torch.zeros((), dtype=A.dtype, device=A.device)

    pfaffian_val = torch.ones((), dtype=A.dtype, device=A.device)
    _A = A.clone()

    for _k in range(0, n - 1, 2):
        full_region = _A[:, _k].clone()
        full_region[:_k+1] = 0.0
        _kp = int(torch.abs(full_region).argmax())

        if _kp != _k + 1:
            _A, sign = pivot(_A, _k, _kp)
        else:
            sign = 1.0

        _A, update = form_gauss_vector(_A, _k)
        pfaffian_val = sign * update * pfaffian_val

    return pfaffian_val

def log_pfaffian_LTL(A):
    """Compute (sign, log|pf|) via Parlett-Reid LTL. Single-matrix, legacy API.
    Delegates to batched implementation.
    """
    return batched_log_pfaffian_ltl(A)
