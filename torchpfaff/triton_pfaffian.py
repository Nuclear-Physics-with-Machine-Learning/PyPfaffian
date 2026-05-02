"""Triton backend for Parlett-Reid LTL Pfaffian.

Parallelizes the batch dimension across Triton program instances, with each
program running the sequential N/2-step LTL decomposition in SRAM.

The pure-PyTorch backend (LogPfaffian in pfaffian.py) is used as fallback
when Triton or CUDA is not available.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _log_pfaffian_ltl_kernel(
    A_ptr,
    sign_ptr,
    logpf_ptr,
    batch_stride,
    row_stride,
    col_stride,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Compute (sign, log|pf|) for one matrix via Parlett-Reid LTL.

    Each program instance handles a single matrix from the batch.
    The full NxN matrix is loaded into a [BLOCK_N, BLOCK_N] register tile
    and the LTL decomposition runs sequentially in N/2 steps.
    """
    pid = tl.program_id(0)

    # Index vectors for rows and columns
    ri = tl.arange(0, BLOCK_N)
    ci = tl.arange(0, BLOCK_N)
    valid = (ri[:, None] < N) & (ci[None, :] < N)

    # Load the full NxN matrix for this batch element
    base = A_ptr + pid * batch_stride
    ptrs = base + ri[:, None] * row_stride + ci[None, :] * col_stride
    W = tl.load(ptrs, mask=valid, other=0.0)

    pf_sign = 1.0
    log_pf = 0.0

    for k in tl.static_range(0, BLOCK_N, 2):
        if k < N - 1:
            # --- Pivot selection ---
            # Extract column k and mask entries at indices <= k
            col_k = tl.sum(tl.where(ci[None, :] == k, W, 0.0), axis=1)
            pivot_mask = ri > k
            abs_col = tl.where(pivot_mask, tl.abs(col_k), 0.0)

            # Argmax: find index of maximum absolute value
            max_val = tl.max(abs_col, axis=0)
            is_max = (abs_col == max_val) & pivot_mask
            # Take the smallest index among ties
            kp = tl.min(tl.where(is_max, ri, N), axis=0)

            k1 = k + 1
            need_swap = kp != k1

            # --- Pivot swap (rows then columns) ---
            # Extract rows k1 and kp
            row_k1 = tl.sum(tl.where(ri[:, None] == k1, W, 0.0), axis=0)
            row_kp = tl.sum(tl.where(ri[:, None] == kp, W, 0.0), axis=0)

            # Swap rows: set row k1 = row_kp, row kp = row_k1
            is_row_k1 = ri[:, None] == k1
            is_row_kp = ri[:, None] == kp
            W = tl.where(is_row_k1 & need_swap, row_kp[None, :], W)
            W = tl.where(is_row_kp & need_swap, row_k1[None, :], W)

            # Extract columns k1 and kp (after row swap)
            col_k1 = tl.sum(tl.where(ci[None, :] == k1, W, 0.0), axis=1)
            col_kp = tl.sum(tl.where(ci[None, :] == kp, W, 0.0), axis=1)

            # Swap columns: set col k1 = col_kp, col kp = col_k1
            is_col_k1 = ci[None, :] == k1
            is_col_kp = ci[None, :] == kp
            W = tl.where(is_col_k1 & need_swap, col_kp[:, None], W)
            W = tl.where(is_col_kp & need_swap, col_k1[:, None], W)

            # Flip sign on swap
            pf_sign = tl.where(need_swap, -pf_sign, pf_sign)

            # --- Gauss vector update ---
            # update = W[k, k+1]
            update = tl.sum(tl.where((ri[:, None] == k) & (ci[None, :] == k1), W, 0.0))

            # mu = W[k, :] / update  (row k divided by pivot)
            mu = tl.sum(tl.where(ri[:, None] == k, W, 0.0), axis=0) / update
            # nu = W[:, k+1]         (column k+1)
            nu = tl.sum(tl.where(ci[None, :] == k1, W, 0.0), axis=1)

            # Rank-2 skew update: W += outer(mu, nu) - outer(nu, mu)
            W = W + mu[:, None] * nu[None, :] - nu[:, None] * mu[None, :]

            # Accumulate sign and log
            pf_sign = pf_sign * tl.where(update > 0, 1.0, -1.0)
            log_pf = log_pf + tl.log(tl.abs(update))

    tl.store(sign_ptr + pid, pf_sign)
    tl.store(logpf_ptr + pid, log_pf)


def triton_log_pfaffian(A):
    """Compute (sign, log|pf(A)|) using the Triton LTL kernel.

    Args:
        A: skew-symmetric matrix of shape (N, N) or (batch, N, N).
           Must be on a CUDA device.

    Returns:
        (sign, log|pf|) tuple with shapes matching the batch dimension.
    """
    squeezed = False
    if A.dim() == 2:
        A = A.unsqueeze(0)
        squeezed = True

    batch, n, _ = A.shape

    if n % 2 == 1:
        sign = torch.ones(batch, dtype=A.dtype, device=A.device)
        logpf = torch.full((batch,), float("-inf"), dtype=A.dtype, device=A.device)
        if squeezed:
            return sign.squeeze(0), logpf.squeeze(0)
        return sign, logpf

    # Ensure contiguous layout for predictable strides
    A = A.contiguous()

    sign = torch.empty(batch, dtype=A.dtype, device=A.device)
    logpf = torch.empty(batch, dtype=A.dtype, device=A.device)

    BLOCK_N = triton.next_power_of_2(n)

    grid = (batch,)
    _log_pfaffian_ltl_kernel[grid](
        A,
        sign,
        logpf,
        A.stride(0),
        A.stride(1),
        A.stride(2),
        N=n,
        BLOCK_N=BLOCK_N,
    )

    if squeezed:
        return sign.squeeze(0), logpf.squeeze(0)
    return sign, logpf
