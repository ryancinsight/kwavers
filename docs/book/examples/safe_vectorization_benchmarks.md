# Example: Safe Vectorization Benchmarks

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example safe_vectorization_benchmarks`  
**Source**: [`crates/kwavers/examples/safe_vectorization_benchmarks.rs`](../../../../crates/kwavers/examples/safe_vectorization_benchmarks.rs)

## What This Example Demonstrates

This example benchmarks safe vectorized kernels against more traditional implementation patterns. The emphasis is on demonstrating that high-level abstractions can still deliver strong throughput and bandwidth for scientific workloads.

| Component | API | Value |
|---|---|---|
| Benchmark record | `BenchmarkResult` | Stores ops/s, bandwidth, relative performance, and numerical error |
| Benchmark suite | `BenchmarkSuite` | Organizes the benchmark categories and collected results |
| Kernel families | array ops / linear algebra / signal processing / physics updates | Covers both general numeric work and simulation-specific kernels |

## Key Code Snippet

```rust
pub struct BenchmarkResult {
    pub test_name: String,
    pub implementation: String,
    pub array_size: usize,
    pub operations_per_second: f64,
    pub memory_bandwidth_gb_per_s: f64,
    pub relative_performance: f64,
    pub accuracy_error: f64,
}
```

## Expected Output (if applicable)

The output is a grouped benchmark report showing throughput, bandwidth, and relative speedups across the tested kernel families.

## Book Chapter

[← Performance and Memory](../performance_and_memory.md)
