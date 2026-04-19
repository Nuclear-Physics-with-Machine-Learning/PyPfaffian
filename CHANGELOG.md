# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-04-18

### Added
- JAX backend: `pfaffian` and `log_pfaffian` with LTL (Parlett-Reid), direct, and recursive methods.
- JAX `custom_jvp` support for differentiation through the Pfaffian.
- Compatible with `jax.jit`, `jax.vmap`, and `jax.grad`.
- PyTorch backend: `pfaffian` and `log_pfaffian` with LTL, direct, and recursive methods.
- PyTorch `torch.autograd.Function` (`LogPfaffian`) with analytical backward pass.
- PyTorch `vmap` batching rule for `LogPfaffian`.
- Triton backend stub for future GPU kernel support.
- Test suite covering correctness, derivatives, batched operations, and cross-backend agreement.

## [0.0.1] - 2024-01-01

### Added
- Initial prototype with JAX Pfaffian implementation.
