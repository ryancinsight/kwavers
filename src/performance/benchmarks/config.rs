//! Benchmark configuration module
//!
//! Provides configuration structures for performance benchmarking.

use std::fmt;

/// Output format for benchmark results
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OutputFormat {
    /// Console output
    Console,
    /// JSON format
    Json,
    /// CSV format
    Csv,
    /// Markdown table
    Markdown,
}

/// Benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Grid sizes to test
    pub grid_sizes: Vec<usize>,
    /// Number of time steps
    pub time_steps: usize,
    /// Number of iterations for averaging
    pub iterations: usize,
    /// Enable GPU benchmarks
    pub enable_gpu: bool,
    /// Enable AMR benchmarks
    pub enable_amr: bool,
    /// Output format
    pub output_format: OutputFormat,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            grid_sizes: vec![64, 128, 256],
            time_steps: 100,
            iterations: 5,
            enable_gpu: true,
            enable_amr: true,
            output_format: OutputFormat::Console,
        }
    }
}

impl BenchmarkConfig {
    /// Create a quick benchmark configuration for testing
    #[must_use]
    pub fn quick() -> Self {
        Self {
            grid_sizes: vec![32, 64],
            time_steps: 10,
            iterations: 2,
            enable_gpu: false,
            enable_amr: false,
            output_format: OutputFormat::Console,
        }
    }

    /// Create a comprehensive benchmark configuration
    #[must_use]
    pub fn comprehensive() -> Self {
        Self {
            grid_sizes: vec![32, 64, 128, 256, 512],
            time_steps: 1000,
            iterations: 10,
            enable_gpu: true,
            enable_amr: true,
            output_format: OutputFormat::Markdown,
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.grid_sizes.is_empty() {
            return Err("Grid sizes cannot be empty".to_string());
        }

        if self.time_steps == 0 {
            return Err("Time steps must be greater than 0".to_string());
        }

        if self.iterations == 0 {
            return Err("Iterations must be greater than 0".to_string());
        }

        for &size in &self.grid_sizes {
            if size == 0 || size > 2048 {
                return Err(format!("Invalid grid size: {size}"));
            }
        }

        Ok(())
    }
}

impl fmt::Display for BenchmarkConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BenchmarkConfig {{ grid_sizes: {:?}, time_steps: {}, iterations: {}, gpu: {}, amr: {} }}",
            self.grid_sizes, self.time_steps, self.iterations, self.enable_gpu, self.enable_amr
        )
    }
}
