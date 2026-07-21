# Example: Adaptive Beamforming Refactored

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example adaptive_beamforming_refactored`  
**Source**: [`crates/kwavers/examples/adaptive_beamforming_refactored.rs`](../../../crates/kwavers/examples/adaptive_beamforming_refactored.rs)

## Overview

Demonstrates the architectural refactoring of the adaptive beamforming module per ADR-001. The monolithic `algorithms_old.rs` (2193 lines, multiple SRP violations) has been replaced with a composable plugin-based structure.

## Architecture achievements

- SRP: each beamforming algorithm is an isolated, testable unit
- Plugin composition via `kwavers_solver::plugin` trait
- Zero-copy field passing via leto `ArrayView`
- Runtime algorithm selection without enum dispatch overhead

## Part Reference

Part II — Transducer Arrays and Beamforming
