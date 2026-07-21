# Example: Monte Carlo Validation

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example monte_carlo_validation`  
**Source**: [`crates/kwavers/examples/monte_carlo_validation.rs`](../../../crates/kwavers/examples/monte_carlo_validation.rs)

## Overview

Compares Monte Carlo photon transport with the diffusion approximation to validate both solvers and understand their respective domains of validity.

## Validation regimes

| Regime | μₐ/μₛ' | Valid solver |
|---|---|---|
| Strongly scattering | < 0.01 | Both (diffusion ≈ MC) |
| Moderate absorption | 0.01–0.1 | MC preferred |
| Highly absorbing | > 0.1 | MC only |

The example sweeps across optical property combinations and computes relative error between the two methods, plotting convergence of the MC solution.

## Part Reference

Part IV — Validation and Benchmarking
