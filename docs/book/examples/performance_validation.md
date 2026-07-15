# Example: Performance Validation

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example performance_validation`  
**Source**: [`crates/kwavers/examples/performance_validation.rs`](../../../../crates/kwavers/examples/performance_validation.rs)

## What This Example Demonstrates

This is a compact performance smoke test for production-readiness discussions. It times foundational operations such as grid allocation, medium construction, large-array creation, and memory accounting on representative problem sizes.

| Component | API | Value |
|---|---|---|
| Baseline grid | `Grid::new(100, 100, 100, ...)` | Benchmarks creation of a 100³ computational domain |
| Medium setup | `HomogeneousMedium::water(&grid)` | Measures initialization cost for a standard acoustic medium |
| Array footprint | `Array3::<f64>::zeros` | Times pressure-field allocation and estimates memory in MB |

## Key Code Snippet

```rust
let grid = Grid::new(100, 100, 100, 1e-3, 1e-3, 1e-3)?;
let grid_creation_time = start.elapsed();
println!(
    "✅ Grid Creation (100³): {:.2}ms",
    grid_creation_time.as_secs_f64() * 1000.0
);

// Test 2: Medium Initialization Performance
let start = Instant::now();
let _medium = HomogeneousMedium::water(&grid);
```

## Expected Output (if applicable)

The console prints millisecond timings for each benchmark stage plus memory-usage and scaling summaries.

## Book Chapter

[← Validation and Benchmarking](../validation_and_benchmarking.md)
