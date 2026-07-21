# Example: Adaptive Beamforming

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example adaptive_beamforming_refactored`  
**Source**: [`crates/kwavers/examples/adaptive_beamforming_refactored.rs`](../../../crates/kwavers/examples/adaptive_beamforming_refactored.rs)

## What This Example Demonstrates

This example is an architecture-focused tour of the adaptive beamforming stack after the ADR-001 refactor. Instead of running a numeric reconstruction, it documents how the monolithic implementation was split into focused submodules while preserving compatibility and test coverage.

| Component | API | Value |
|---|---|---|
| Architecture | `adaptive_beamforming::{adaptive, conventional, subspace, tapering, past, opast}` | Focused modules replace the old 2193-line implementation |
| API ownership | `kwavers_analysis::...::adaptive::MinimumVariance` | Analysis owns adaptive weighting; transducer owns hardware interfaces |
| Verification | Package Nextest and strict Clippy | Validates the current source graph |

## Key Code Snippet

```rust
println!("Adaptive Beamforming - Architecture Refactoring Complete");
println!("=======================================================");

println!("\n✓ REFACTORING ACHIEVEMENTS:");
println!("  • Eliminated monolithic algorithms_old.rs (2193 lines)");
println!("  • Split into focused submodules (<500 lines each)");
println!("  • Removed code duplication across algorithms");
println!("  • Feature-gated legacy implementations");
println!("  • Maintained 100% backwards compatibility");
```

## Expected Output (if applicable)

The executable prints refactoring achievements, quality-assurance checks, and the new module layout rather than generating beamformed images.

## Book Chapter

[← Transducer Arrays and Beamforming](../beamforming_and_image_formation.md)
