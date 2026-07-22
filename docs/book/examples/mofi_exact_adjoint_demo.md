# Example: MoFi Exact Adjoint Demo

**Crate**: `kwavers-solver`
**Run**: `cargo run -p kwavers-solver --example mofi_exact_adjoint_demo`
**Source**: [`crates/kwavers-solver/examples/mofi_exact_adjoint_demo.rs`](../../../crates/kwavers-solver/examples/mofi_exact_adjoint_demo.rs)

## What This Example Demonstrates

Guidance-free skull-template alignment using the exact-adjoint engine (MOFI). Builds an asymmetric 2D sound-speed template, applies a known rigid misalignment, generates acoustic data with the self-adjoint exact-gradient engine (`FwiEngine::SecondOrderSelfAdjoint`, ADR 016), and recovers the pose by manifold optimisation of the acoustic misfit alone — no guidance image.

## Key Concepts

- Self-adjoint exact-gradient FWI engine (ADR 016)
- Rigid-body manifold optimisation (translation + rotation)
- Guidance-free skull-template registration from acoustic data alone
