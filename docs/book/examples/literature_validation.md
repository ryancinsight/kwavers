# Example: Literature Validation

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example literature_validation_safe`  
**Source**: [`crates/kwavers/examples/literature_validation_safe.rs`](../../../../crates/kwavers/examples/literature_validation_safe.rs)

## What This Example Demonstrates

This validation example packages canonical acoustic benchmarks from the literature and checks both numerical accuracy and safe-vectorization performance. The selected cases span Green's functions, diffraction, interference, attenuation, and nonlinear wave evolution.

| Component | API | Value |
|---|---|---|
| Result record | `ValidationResult` | Captures max error, RMS error, timing, pass/fail state, and notes |
| Numerical domain | `Grid::new(nx, ny, nz, dx, dx, dx)` | Creates the shared grid used by the validation kernels |
| Reference set | Pierce / Born & Wolf / Kinsler / Blackstock / Hamilton-Blackstock | Provides published targets and expected error tolerances |

## Key Code Snippet

```rust
pub struct ValidationResult {
    pub test_name: String,
    pub reference: String,
    pub max_error: f64,
    pub rms_error: f64,
    pub computation_time: f64,
    pub passed: bool,
    pub details: String,
}
```

## Expected Output (if applicable)

The executable reports one validation result per literature case, including numerical error and computation time.

## Book Chapter

[← Validation and Benchmarking](../validation_and_benchmarking.md)
