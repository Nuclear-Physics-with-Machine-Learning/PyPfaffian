# torchpfaff Benchmarks

## Quick start

```bash
# GPU (CUDA) benchmark, float64
python benchmarks/benchmark.py --device cuda --dtype float64

# CPU-only, subset of sizes
python benchmarks/benchmark.py --device cpu --N 4 8 16 --batch 256 1024

# Save results to CSV for plotting
python benchmarks/benchmark.py --device cuda --csv results.csv
```

## CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--device` | `cuda` if available | `cpu` or `cuda` |
| `--dtype` | `float64` | `float32` or `float64` |
| `--N` | 2 4 6 8 12 16 20 32 40 64 | Matrix sizes to benchmark |
| `--batch` | 1 32 256 1024 4096 | Batch sizes |
| `--warmup` | 3 | Warmup iterations (not timed) |
| `--repeats` | 20 | Timed iterations (median reported) |
| `--csv` | none | Write results to a CSV file |
| `--skip-correctness` | off | Skip the pf^2==det sanity check |

## What is measured

The benchmark times `log_pfaffian(A, backend=...)` for batches of random
skew-symmetric matrices.  For each configuration it reports:

- **Median(ms)** — median wall time across `--repeats` runs
- **Mat/s** — throughput in matrices per second (batch / time)
- **GB/s** — effective memory bandwidth (input bytes / time)
- **GFLOP/s** — achieved floating-point throughput
- **SoL%** — speed-of-light percentage (see below)

## Speed-of-light analysis

The Parlett-Reid LTL decomposition for a batch of B matrices of size N has:

- **FLOPs**: `2 * B * N^3` (leading order — N/2 steps, each doing a rank-2 update over N^2 elements)
- **Bytes**: `B * N^2 * sizeof(element)` input read + `2 * B * sizeof(element)` output written

The **arithmetic intensity** is `2*N / sizeof(element)` FLOPs/byte:

| N | float32 (4B) | float64 (8B) |
|---|-------------|-------------|
| 4 | 2.0 | 1.0 |
| 8 | 4.0 | 2.0 |
| 16 | 8.0 | 4.0 |
| 32 | 16.0 | 8.0 |
| 64 | 32.0 | 16.0 |

Modern GPUs have a ridge point around 100-200 FLOPs/byte (A100: ~150, H100: ~220).
For the typical QMC matrix sizes (N=4..40), the kernel is firmly in the
**memory-bandwidth-bound** regime.

**SoL%** = `(theoretical_minimum_time / measured_time) * 100`, where the
theoretical minimum time assumes data is loaded at peak memory bandwidth.

- **SoL% > 100%** can happen when data fits in L2 cache, bypassing DRAM
  bandwidth limits.  This is expected for small matrices.
- **SoL% ~ 50-80%** is good for a real kernel with pivot branching.
- **SoL% < 10%** indicates the backend is overhead-dominated (e.g. Python
  loop + kernel launch overhead in the PyTorch backend at small N).

## Interpreting results

**Triton vs PyTorch-CUDA**: The Triton kernel fuses the entire LTL
decomposition into a single kernel launch per batch element, eliminating
the N/2 Python-loop iterations and ~6 kernel launches per step that the
PyTorch backend requires.  Expect Triton to be dramatically faster at
small N and large B.

**Large N**: As N grows, the rank-2 update (`bmm`) in the PyTorch backend
becomes a larger fraction of total time and kernel launch overhead shrinks
proportionally.  The two backends converge.

**CPU**: The Python loop overhead is less severe on CPU (no kernel launch
cost), but the single-threaded sequential algorithm doesn't benefit from
CPU parallelism.  CPU numbers are mainly useful as a baseline.
