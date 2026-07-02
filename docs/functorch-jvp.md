# Functional VJP/JVP support for `LogPfaffian`

Branch: `feat/functorch-jvp`

Status: full CPU suite passes; CUDA/Triton functional-transform validation
pending.

## Motivation

`LogPfaffian` already supplied an analytical reverse-mode derivative and an
explicit `vmap` rule.  Two functional compositions were still unavailable:

1. `torch.func.jvp`/`jacfwd` failed because the custom autograd Function did
   not implement `jvp`.
2. A VJP of batched outputs, conceptually `grad(vmap(LogPfaffian))`, failed
   because the `vmap` rule returned the raw `_log_pfaffian_op`.  That bypassed
   the outer `LogPfaffian` autograd Function, whose analytical backward is what
   makes the operation differentiable.

Per-example `vmap(grad(LogPfaffian))` worked, but it is not enough for a true
weighted-batch backward or a matrix-free parameter-Jacobian operator.  The
matrix-free MinSR branch requires both `J^T v` and `J u` without materializing
per-walker parameter gradients.

## Changes

### Preserve autograd under `vmap`

The batching rule still unwraps/moves the physical batch dimension, but now
re-enters

```python
LogPfaffian.apply(A)
```

instead of calling `_log_pfaffian_op(A)` directly.  The tensor passed to the
batching-rule body is already a physical, unwrapped tensor, so this does not
recurse into the same `vmap` rule.  It preserves the outer autograd Function
and therefore its analytical backward under an enclosing VJP.

### Save state for forward mode

`setup_context` now saves `A` in both stores:

- `save_for_backward(A)` for reverse mode;
- `save_for_forward(A)` for `jvp`.

### Analytical JVP

For a nonsingular skew-symmetric matrix,

```text
d log|pf(A)| = -1/2 sum_ij (A^-1)_ij dA_ij.
```

The new rule contracts this gradient with the incoming matrix tangent.  The
Pfaffian sign is piecewise constant, so its tangent is zero:

```text
sign_dot  = 0
logpf_dot = sum(-0.5 * inv(A) * A_dot).
```

This is the same convention used by the existing analytical backward.

## Compatibility

- The public `log_pfaffian` and `LogPfaffian.apply` APIs are unchanged.
- The forward kernel and Triton dispatch are unchanged.
- No parameter, state-dict, serialization, or numerical forward format changes.
- Existing reverse-mode and `vmap(grad(...))` users continue through the same
  analytical derivative.
- `log_pfaffian` remains a Dynamo graph-break boundary because the upstream
  custom-op/autograd interaction described in its source comments still
  applies.  This branch adds functional derivatives; it does not remove that
  compile workaround.

## Implementation

Files:

- `torchpfaff/pfaffian.py`;
- `tests/torch/test_derivative.py`.

The derivative tests now treat `jacfwd` and `torch.func.jvp` as supported
instead of expected failures.  A new weighted VJP regression applies a
pullback to `vmap(LogPfaffian)` and compares it with the batched analytical
inverse formula.

## CPU validation

The full `tests/torch` suite passed at implementation time:

```text
444 passed, 140 skipped, 2 existing xfailed
```

The compile-focused subset also passed its CPU cases.  In an additional full
`torch_qmc` Pfaffian model probe:

- a weighted batched parameter VJP matched the dense `J^T v` to about `7e-14`;
- a parameter JVP matched dense `J u` to about `2e-12`.

These full-model products are also exercised by the matrix-free MinSR branch's
tests.

## CUDA validation

On a GPU system, run the normal PyPfaffian suite including CUDA tests and add a
representative `torch_qmc` check with the production Pfaffian size:

1. eager batched forward and reverse mode;
2. weighted VJP of batched outputs;
3. parameter JVP through the full wavefunction;
4. `torch.compile` around the surrounding model with the expected Pfaffian
   graph break;
5. Triton versus PyTorch backend parity for values and both derivative
   products;
6. repeated VJP/JVP calls to check memory stability.

## Known limitations

- The derivative assumes a nonsingular matrix; behavior near a vanishing
  Pfaffian remains governed by the matrix inverse, as in the existing backward.
- The JVP computes an inverse independently.  A future optimization could
  reuse a factorization when the forward backend exposes one.
- The sign tangent is zero away from sign-changing singularities and is not
  defined at a zero Pfaffian.
- This branch is a prerequisite for Pfaffian support in matrix-free MinSR; the
  optimizer branch should not be deployed with the unpatched dependency.
