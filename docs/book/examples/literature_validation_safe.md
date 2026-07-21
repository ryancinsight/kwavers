# Example: Literature Validation with Safe Vectorization

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example literature_validation_safe`  
**Source**: [`crates/kwavers/examples/literature_validation_safe.rs`](../../../crates/kwavers/examples/literature_validation_safe.rs)

## Overview

Reproduces well-known test cases from the acoustic wave propagation literature using the safe vectorization path (`hermes_simd` runtime dispatch). Compares accuracy and performance against the traditional unsafe SIMD approach.

## Test cases

- Gaussian pulse propagation in homogeneous medium (CFL stability check)
- Plane wave reflection coefficient at fluid–fluid interface
- Focused transducer pressure field (Rayleigh integral validation)

## Key result

Safe vectorization matches unsafe SIMD accuracy to machine precision while remaining `#[forbid(unsafe_code)]`-compliant.

## Part Reference

Part IV — Validation and Benchmarking
