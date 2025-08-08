//! Benchmarking Infrastructure
//!
//! This module provides comprehensive benchmarking capabilities for
//! performance testing and optimization.

pub mod accuracy;
pub mod suite;

pub use accuracy::{AccuracyBenchmark, AccuracyMetrics};
pub use suite::{BenchmarkSuite, BenchmarkConfig, BenchmarkReport, OutputFormat};

#[cfg(test)]
mod tests {
    
    
    #[test]
    fn test_benchmark_framework() {
        // Simple test to ensure benchmark framework compiles
        assert!(true);
    }
}