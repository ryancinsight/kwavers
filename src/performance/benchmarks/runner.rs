//! Benchmark runner implementation
//!
//! Orchestrates the execution of various benchmarks.

use super::config::BenchmarkConfig;
use super::result::{BenchmarkReport, BenchmarkResult};
use crate::{Grid, KwaversResult};
use std::time::Instant;

/// Benchmark runner
pub struct BenchmarkRunner {
    config: BenchmarkConfig,
    report: BenchmarkReport,
}

impl BenchmarkRunner {
    /// Create a new benchmark runner
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            report: BenchmarkReport::new(),
        }
    }

    /// Run all enabled benchmarks
    pub fn run_all(&mut self) -> KwaversResult<BenchmarkReport> {
        // Validate configuration
        self.config
            .validate()
            .map_err(crate::error::KwaversError::InvalidInput)?;

        log::info!("Starting benchmark suite with config: {}", self.config);

        // Run FDTD benchmarks
        self.run_fdtd_benchmarks()?;

        // Run PSTD benchmarks
        self.run_pstd_benchmarks()?;

        // Run GPU benchmarks if enabled
        if self.config.enable_gpu {
            self.run_gpu_benchmarks()?;
        }

        // Run AMR benchmarks if enabled
        if self.config.enable_amr {
            self.run_amr_benchmarks()?;
        }

        Ok(self.report.clone())
    }

    /// Run FDTD benchmarks
    fn run_fdtd_benchmarks(&mut self) -> KwaversResult<()> {
        for &grid_size in &self.config.grid_sizes {
            let result = super::fdtd::benchmark_fdtd(
                grid_size,
                self.config.time_steps,
                self.config.iterations,
            )?;
            self.report.add_result(result);
        }
        Ok(())
    }

    /// Run PSTD benchmarks
    fn run_pstd_benchmarks(&mut self) -> KwaversResult<()> {
        for &grid_size in &self.config.grid_sizes {
            let result = super::pstd::benchmark_pstd(
                grid_size,
                self.config.time_steps,
                self.config.iterations,
            )?;
            self.report.add_result(result);
        }
        Ok(())
    }

    /// Run GPU benchmarks
    fn run_gpu_benchmarks(&mut self) -> KwaversResult<()> {
        #[cfg(feature = "gpu")]
        {
            for &grid_size in &self.config.grid_sizes {
                let result = super::gpu::benchmark_gpu(
                    grid_size,
                    self.config.time_steps,
                    self.config.iterations,
                )?;
                self.report.add_result(result);
            }
        }
        #[cfg(not(feature = "gpu"))]
        {
            log::warn!("GPU benchmarks requested but GPU feature not enabled");
        }
        Ok(())
    }

    /// Run AMR benchmarks
    fn run_amr_benchmarks(&mut self) -> KwaversResult<()> {
        for &grid_size in &self.config.grid_sizes {
            let result = super::amr::benchmark_amr(
                grid_size,
                self.config.time_steps,
                self.config.iterations,
            )?;
            self.report.add_result(result);
        }
        Ok(())
    }

    /// Run a custom benchmark
    pub fn run_custom<F>(
        &mut self,
        name: &str,
        grid_size: usize,
        benchmark_fn: F,
    ) -> KwaversResult<()>
    where
        F: Fn(&Grid) -> KwaversResult<()>,
    {
        let grid = Grid::new(grid_size, grid_size, grid_size, 1e-3, 1e-3, 1e-3);
        let mut times = Vec::with_capacity(self.config.iterations);

        // Warmup
        benchmark_fn(&grid)?;

        // Actual benchmarking
        for _ in 0..self.config.iterations {
            let start = Instant::now();
            benchmark_fn(&grid)?;
            times.push(start.elapsed());
        }

        let result = BenchmarkResult::new(name.to_string(), grid_size, times);
        self.report.add_result(result);

        Ok(())
    }

    /// Get the current report
    pub fn report(&self) -> &BenchmarkReport {
        &self.report
    }

    /// Print results to console
    pub fn print_results(&self) {
        println!("{}", self.report.summary());
    }
}
