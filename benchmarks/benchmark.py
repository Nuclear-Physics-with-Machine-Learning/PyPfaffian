#!/usr/bin/env python3
"""torchpfaff benchmark suite with speed-of-light analysis.

Measures log_pfaffian throughput across matrix sizes, batch sizes, dtypes,
and backends (PyTorch CPU, PyTorch CUDA, Triton).  Computes effective
bandwidth and GFLOP/s, and compares against theoretical memory-bandwidth
speed-of-light for the Parlett-Reid LTL algorithm.

Usage:
    python benchmarks/benchmark.py --device cuda --dtype float64
    python benchmarks/benchmark.py --device cpu --dtype float64 --N 4 8 16 --batch 256 1024
    python benchmarks/benchmark.py --device cuda --csv results.csv
"""

import argparse
import csv
import statistics
import sys
import time

import torch

from torchpfaff import log_pfaffian

_triton_available = False
try:
    import triton  # noqa: F401

    _triton_available = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Known peak memory bandwidths (GB/s) for common GPUs.
# Used when torch can't report it directly.
# ---------------------------------------------------------------------------
_KNOWN_BW = {
    "A100-SXM4-40GB": 2039,
    "A100-SXM4-80GB": 2039,
    "A100-PCIE-40GB": 1555,
    "A100-PCIE-80GB": 1935,
    "H100-SXM": 3350,
    "H100-PCIE": 2039,
    "V100-SXM2": 900,
    "V100-PCIE": 900,
    "A6000": 768,
    "RTX 4090": 1008,
    "RTX 3090": 936,
    "GH200": 4800,
}


def get_peak_bandwidth_gb(device: torch.device) -> float | None:
    """Return estimated peak memory bandwidth in GB/s, or None if unknown."""
    if device.type != "cuda":
        return None
    props = torch.cuda.get_device_properties(device)
    name = props.name

    for key, bw in _KNOWN_BW.items():
        if key.lower().replace("-", " ") in name.lower().replace("-", " "):
            return float(bw)

    # Fallback: estimate from memory clock and bus width if available.
    # torch exposes total_memory but not clock/bus, so just return None.
    return None


def make_skew_symmetric(
    B: int, N: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Generate a batch of random skew-symmetric matrices."""
    M = torch.randn(B, N, N, dtype=dtype, device=device)
    return M - M.transpose(-1, -2)


# ---------------------------------------------------------------------------
# FLOP and byte model for Parlett-Reid LTL
# ---------------------------------------------------------------------------


def ltl_flops(B: int, N: int) -> int:
    """Leading-order FLOPs for batched LTL: 2*B*N^3."""
    return 2 * B * N**3


def ltl_bytes(B: int, N: int, dtype: torch.dtype) -> int:
    """Minimum bytes transferred: read B*N*N input + write 2*B scalars."""
    elem = torch.finfo(dtype).bits // 8
    return B * N * N * elem + 2 * B * elem


def sol_time_s(B: int, N: int, dtype: torch.dtype, peak_bw_gb: float) -> float:
    """Bandwidth-bound speed-of-light time in seconds."""
    return ltl_bytes(B, N, dtype) / (peak_bw_gb * 1e9)


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------


def check_correctness(A: torch.Tensor, backend: str) -> bool:
    """Verify pf(A)^2 == det(A) for the first matrix in the batch."""
    A0 = A[0:1]
    sign, logpf = log_pfaffian(A0, backend=backend)
    pf_sq = torch.exp(2 * logpf)
    det = torch.linalg.det(A0).abs()
    return torch.allclose(pf_sq, det, rtol=1e-4, atol=1e-6)


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _time_cpu(A: torch.Tensor, backend: str, warmup: int, repeats: int) -> list[float]:
    """Time log_pfaffian on CPU, return list of elapsed seconds."""
    for _ in range(warmup):
        log_pfaffian(A, backend=backend)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        log_pfaffian(A, backend=backend)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def _time_cuda(A: torch.Tensor, backend: str, warmup: int, repeats: int) -> list[float]:
    """Time log_pfaffian on CUDA using cuda Events, return list of seconds."""
    for _ in range(warmup):
        log_pfaffian(A, backend=backend)
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        log_pfaffian(A, backend=backend)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1000.0)  # ms -> s
    return times


def bench(A: torch.Tensor, backend: str, warmup: int, repeats: int) -> float:
    """Return median wall time in seconds."""
    if A.is_cuda:
        times = _time_cuda(A, backend, warmup, repeats)
    else:
        times = _time_cpu(A, backend, warmup, repeats)
    return statistics.median(times)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="torchpfaff benchmark with speed-of-light analysis"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Device to benchmark on (default: cuda if available)",
    )
    parser.add_argument(
        "--dtype",
        default="float64",
        choices=["float32", "float64"],
        help="Data type (default: float64)",
    )
    parser.add_argument(
        "--N",
        type=int,
        nargs="+",
        default=[2, 4, 6, 8, 12, 16, 20, 32, 40, 64],
        help="Matrix sizes to benchmark",
    )
    parser.add_argument(
        "--batch",
        type=int,
        nargs="+",
        default=[1, 32, 256, 1024, 4096],
        help="Batch sizes to benchmark",
    )
    parser.add_argument(
        "--warmup", type=int, default=3, help="Number of warmup iterations (default: 3)"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=20,
        help="Number of timed iterations (default: 20)",
    )
    parser.add_argument(
        "--csv", type=str, default=None, help="Write results to CSV file"
    )
    parser.add_argument(
        "--backend",
        type=str,
        nargs="+",
        default=None,
        choices=["pytorch", "triton"],
        help="Backends to benchmark (default: all available)",
    )
    parser.add_argument(
        "--skip-correctness",
        action="store_true",
        help="Skip pf^2==det correctness check",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    # Determine backends to run
    if args.backend is not None:
        backends = args.backend
        if "triton" in backends and not _triton_available:
            print(
                "ERROR: triton backend requested but triton is not installed",
                file=sys.stderr,
            )
            sys.exit(1)
        if "triton" in backends and device.type != "cuda":
            print("ERROR: triton backend requires --device cuda", file=sys.stderr)
            sys.exit(1)
    elif device.type == "cpu":
        backends = ["pytorch"]
    else:
        backends = ["pytorch"]
        if _triton_available:
            backends.append("triton")

    # Header
    peak_bw: float | None = None
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_properties(device).name
        peak_bw = get_peak_bandwidth_gb(device)
        print(f"torchpfaff Benchmark — device: {gpu_name}, dtype: {args.dtype}")
        if peak_bw is not None:
            print(f"Peak Memory BW: {peak_bw:.0f} GB/s")
        else:
            print("Peak Memory BW: unknown (SoL% will be omitted)")
    else:
        print(f"torchpfaff Benchmark — device: CPU, dtype: {args.dtype}")
    print()

    # Table header
    hdr_sol = "SoL%" if peak_bw is not None else ""
    header = (
        f"{'N':>4s}  {'Batch':>6s}  {'Backend':<14s}  "
        f"{'Median(ms)':>10s}  {'Mat/s':>10s}  "
        f"{'GB/s':>8s}  {'GFLOP/s':>8s}"
    )
    if hdr_sol:
        header += f"  {hdr_sol:>6s}"
    print(header)
    print("-" * len(header))

    rows: list[dict] = []

    for N in args.N:
        for B in args.batch:
            A = make_skew_symmetric(B, N, dtype, device)

            for backend in backends:
                # Correctness check (once per config)
                if not args.skip_correctness and N >= 2 and N % 2 == 0:
                    ok = check_correctness(A, backend)
                    if not ok:
                        print(
                            f"  *** CORRECTNESS FAILURE: N={N} B={B} "
                            f"backend={backend} ***"
                        )

                median_s = bench(A, backend, args.warmup, args.repeats)
                median_ms = median_s * 1000

                mats_per_s = B / median_s
                flops = ltl_flops(B, N)
                nbytes = ltl_bytes(B, N, dtype)
                gflops = flops / median_s / 1e9
                gbps = nbytes / median_s / 1e9

                # Format throughput with suffix
                if mats_per_s >= 1e6:
                    mats_str = f"{mats_per_s / 1e6:.2f}M"
                elif mats_per_s >= 1e3:
                    mats_str = f"{mats_per_s / 1e3:.2f}K"
                else:
                    mats_str = f"{mats_per_s:.1f}"

                label = f"pytorch-{device.type}" if backend == "pytorch" else backend
                line = (
                    f"{N:4d}  {B:6d}  {label:<14s}  "
                    f"{median_ms:10.3f}  {mats_str:>10s}  "
                    f"{gbps:8.2f}  {gflops:8.2f}"
                )

                sol_pct = None
                if peak_bw is not None and device.type == "cuda":
                    sol = sol_time_s(B, N, dtype, peak_bw)
                    sol_pct = sol / median_s * 100
                    line += f"  {sol_pct:5.1f}%"

                print(line)

                rows.append(
                    {
                        "N": N,
                        "batch": B,
                        "backend": label,
                        "dtype": args.dtype,
                        "median_ms": round(median_ms, 4),
                        "mat_per_s": round(mats_per_s, 1),
                        "gbps": round(gbps, 3),
                        "gflops": round(gflops, 3),
                        "sol_pct": round(sol_pct, 2) if sol_pct is not None else "",
                    }
                )

            del A
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # CSV output
    if args.csv and rows:
        fieldnames = [
            "N",
            "batch",
            "backend",
            "dtype",
            "median_ms",
            "mat_per_s",
            "gbps",
            "gflops",
            "sol_pct",
        ]
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults written to {args.csv}")


if __name__ == "__main__":
    main()
