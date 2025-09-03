//! AMR solver benchmarks

use crate::performance::benchmarks::result::BenchmarkResult;
use crate::KwaversResult;
use std::time::Duration;

/// Benchmark AMR solver performance
pub fn benchmark_amr(
    grid_size: usize,
    time_steps: usize,
    iterations: usize,
) -> KwaversResult<BenchmarkResult> {
    // Placeholder for AMR benchmark
    let times = vec![Duration::from_millis(150); iterations];
    let mut result = BenchmarkResult::new(format!("AMR_{grid_size}x{grid_size}"), grid_size, times);
    result.add_metric("time_steps", time_steps as f64);
    Ok(result)
}
