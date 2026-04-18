"""Triton backend for Parlett-Reid LTL Pfaffian.

This module is a placeholder. The Triton kernel implementation requires
a CUDA GPU and the triton package. When available, it parallelizes the
batch dimension across Triton program instances, with each program running
the sequential N/2-step LTL decomposition in SRAM.

The pure-PyTorch backend (LogPfaffian in pfaffian.py) is used as fallback
when Triton or CUDA is not available.
"""

def triton_log_pfaffian(A):
    raise NotImplementedError(
        "Triton Pfaffian backend is not yet implemented. "
        "Use backend='pytorch' or ensure the auto-dispatch selects "
        "the PyTorch backend (which happens automatically on CPU)."
    )
