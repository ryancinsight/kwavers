//! GPU solver benchmarks

use crate::performance::benchmarks::result::BenchmarkResult;
use crate::KwaversResult;
use std::time::Duration;

/// Benchmark GPU solver performance
#[cfg(feature = "gpu")]
pub fn benchmark_gpu(
    grid_size: usize,
    time_steps: usize,
    iterations: usize,
) -> KwaversResult<BenchmarkResult> {
    // Placeholder for GPU benchmark
    let times = vec![Duration::from_millis(50); iterations];
    let mut result =
        BenchmarkResult::new(format!("GPU_{}x{}", grid_size, grid_size), grid_size, times);
    result.add_metric("time_steps", time_steps as f64);
    Ok(result)
}

#[cfg(not(feature = "gpu"))]
pub fn benchmark_gpu(
    grid_size: usize,
    _time_steps: usize,
    iterations: usize,
) -> KwaversResult<BenchmarkResult> {
    let times = vec![Duration::from_millis(0); iterations];
    Ok(BenchmarkResult::new(
        format!("GPU_{}x{}_disabled", grid_size, grid_size),
        grid_size,
        times,
    ))
}
