//! Benchmarking Infrastructure
//!
//! This module provides comprehensive benchmarking capabilities for
//! performance testing and optimization.

pub mod accuracy;
pub mod suite;

pub use accuracy::AccuracyResult;
pub use suite::{BenchmarkConfig, BenchmarkReport, BenchmarkSuite, OutputFormat};

#[cfg(test)]
mod tests {

    #[test]
    fn test_benchmark_framework() {
        // Simple test to ensure benchmark framework compiles
        assert!(true);
    }
}
