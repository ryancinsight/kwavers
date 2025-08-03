//! Performance Optimization Module
//!
//! This module provides advanced performance optimization techniques
//! to achieve 100M+ grid updates per second.

pub mod optimization;

pub use optimization::{
    PerformanceOptimizer,
    OptimizationConfig,
    SimdLevel,
    StencilKernel,
    BandwidthOptimizer,
    CacheOptimizer,
    AccessPattern,
    PrefetchStrategy,
};