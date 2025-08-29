//! Performance benchmarking module
//!
//! Provides comprehensive benchmarking capabilities for the Kwavers framework.

pub mod config;
pub mod result;
pub mod runner;

// Benchmark implementations
pub mod amr;
pub mod fdtd;
pub mod gpu;
pub mod pstd;

// Re-exports
pub use config::{BenchmarkConfig, OutputFormat};
pub use result::{BenchmarkReport, BenchmarkResult};
pub use runner::BenchmarkRunner;

use crate::KwaversResult;

/// Run all benchmarks with default configuration
pub fn run_all() -> KwaversResult<BenchmarkReport> {
    let config = BenchmarkConfig::default();
    let mut runner = BenchmarkRunner::new(config);
    runner.run_all()
}

/// Run quick benchmarks for CI/CD
pub fn run_quick() -> KwaversResult<BenchmarkReport> {
    let config = BenchmarkConfig::quick();
    let mut runner = BenchmarkRunner::new(config);
    runner.run_all()
}

/// Run comprehensive benchmarks for detailed analysis
pub fn run_comprehensive() -> KwaversResult<BenchmarkReport> {
    let config = BenchmarkConfig::comprehensive();
    let mut runner = BenchmarkRunner::new(config);
    runner.run_all()
}
