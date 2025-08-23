//! Performance Optimization Module
//!
//! This module provides performance optimization techniques
//! to achieve 100M+ grid updates per second.
//!
//! ## Literature References
//!
//! 1. **Williams, S., Waterman, A., & Patterson, D. (2009)**. "Roofline: an
//!    insightful visual performance model for multicore architectures."
//!    *Communications of the ACM*, 52(4), 65-76. DOI: 10.1145/1498765.1498785
//!    - Roofline model for performance analysis
//!    - Bandwidth and compute bounds
//!
//! 2. **Datta, K., et al. (2008)**. "Stencil computation optimization and
//!    auto-tuning on state-of-the-art multicore architectures." *SC'08:
//!    Proceedings of the 2008 ACM/IEEE conference on Supercomputing* (pp. 1-12).
//!    DOI: 10.1109/SC.2008.5222004
//!    - Cache blocking strategies
//!    - SIMD optimization techniques
//!
//! 3. **Kamil, S., et al. (2010)**. "An auto-tuning framework for parallel
//!    multicore stencil computations." *2010 IEEE International Symposium on
//!    Parallel & Distributed Processing* (pp. 1-12). DOI: 10.1109/IPDPS.2010.5470421
//!    - Auto-tuning strategies
//!    - Performance portability
//!
//! ## Design Principles
//! - **Zero-Copy**: Extensive use of slices and views
//! - **KISS**: Simple profiling interface with powerful insights
//! - **DRY**: Reusable performance patterns
//! - **YAGNI**: Only essential profiling features

pub mod benchmarks;
pub mod metrics;
pub mod optimization;
pub mod profiling;
pub mod simd;

pub use optimization::{
    AccessPattern, BandwidthOptimizer, CacheOptimizer, OptimizationConfig, PerformanceOptimizer,
    PrefetchStrategy, SimdLevel, StencilKernel,
};

pub use profiling::{
    CacheProfile, MemoryEventType, MemoryProfile, PerformanceBound, PerformanceProfiler,
    ProfileReport, RooflineAnalysis, TimingScope,
};

// Re-export benchmarking functionality
pub use benchmarks::{
    AccuracyResult, BenchmarkConfig, BenchmarkReport, BenchmarkSuite, OutputFormat,
};
