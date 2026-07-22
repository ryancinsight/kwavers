# Example: Transcranial UST Reconstruction

**Crate**: `kwavers-solver`
**Run**: `cargo run -p kwavers-solver --example transcranial_ust_reconstruction`
**Source**: [`crates/kwavers-solver/examples/transcranial_ust_reconstruction.rs`](../../../crates/kwavers-solver/examples/transcranial_ust_reconstruction.rs)

## What This Example Demonstrates

Transcranial-UST reconstruction figure data: a 2D head phantom (skull annulus + brain structures), a known rigid misalignment of the CT template, and the MOFI guidance-free alignment recovered from the acoustic data alone with the exact-adjoint engine (ADR 016/017).

## Outputs

CSV grids (patient / misaligned-template / MOFI-aligned / error) to `target/book_data/transcranial/`, consumed by the Python figure script `crates/kwavers-python/examples/book/ch26_transcranial_reconstruction.py`.
