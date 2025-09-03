//! PSTD solver benchmarks

use crate::performance::benchmarks::result::BenchmarkResult;
use crate::KwaversResult;
use std::time::Duration;

/// Benchmark PSTD solver performance
pub fn benchmark_pstd(
    grid_size: usize,
    time_steps: usize,
    iterations: usize,
) -> KwaversResult<BenchmarkResult> {
    // Placeholder for PSTD benchmark
    let times = vec![Duration::from_millis(100); iterations];
    let mut result =
        BenchmarkResult::new(format!("PSTD_{grid_size}x{grid_size}"), grid_size, times);
    result.add_metric("time_steps", time_steps as f64);
    Ok(result)
}
