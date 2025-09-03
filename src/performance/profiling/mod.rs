//! Performance profiling infrastructure
//!
//! This module provides comprehensive performance profiling capabilities
//! including timing, memory usage, cache behavior, and roofline analysis.

pub mod analysis;
pub mod cache;
pub mod memory;
pub mod timing;

// Re-export main types
pub use analysis::{PerformanceAnalyzer, PerformanceBound, RooflineAnalysis};
pub use cache::{CacheProfile, CacheProfiler, CacheStatistics};
pub use memory::{MemoryEvent, MemoryEventType, MemoryProfile, MemoryProfiler};
pub use timing::{TimingProfiler, TimingScope, TimingSummary};

use crate::grid::Grid;
use std::time::Duration;

/// Comprehensive performance profiler
#[derive(Debug)]
pub struct PerformanceProfiler {
    /// Timing profiler
    pub timing: TimingProfiler,
    /// Memory profiler
    pub memory: MemoryProfiler,
    /// Cache profiler
    pub cache: CacheProfiler,
    /// Performance analyzer
    pub analyzer: PerformanceAnalyzer,
}

impl PerformanceProfiler {
    /// Create a new performance profiler
    pub fn new(grid: Grid) -> Self {
        Self {
            timing: TimingProfiler::new(),
            memory: MemoryProfiler::new(),
            cache: CacheProfiler::new(),
            analyzer: PerformanceAnalyzer::new(grid),
        }
    }

    /// Start a timing scope
    pub fn scope(&self, name: &str) -> TimingScope {
        self.timing.scope(name)
    }

    /// Record memory allocation
    pub fn allocate(&self, bytes: usize) {
        self.memory.allocate(bytes);
    }

    /// Record memory deallocation
    pub fn deallocate(&self, bytes: usize) {
        self.memory.deallocate(bytes);
    }

    /// Clear all profiling data
    pub fn clear(&self) {
        self.timing.clear();
        self.memory.clear();
        self.cache.clear();
    }

    /// Generate comprehensive report
    pub fn report(&self) -> ProfileReport {
        ProfileReport {
            timing_summaries: self.timing.summaries(),
            memory_profile: self.memory.profile(),
            cache_profile: self.cache.profile(),
            performance_analysis: self.analyzer.report(),
        }
    }
}

/// Comprehensive profile report
#[derive(Debug, Clone)]
pub struct ProfileReport {
    /// Timing summaries
    pub timing_summaries: Vec<TimingSummary>,
    /// Memory profile
    pub memory_profile: MemoryProfile,
    /// Cache profile
    pub cache_profile: CacheProfile,
    /// Performance analysis
    pub performance_analysis: String,
}

impl ProfileReport {
    /// Generate text report
    #[must_use]
    pub fn to_string(&self) -> String {
        let mut report = String::new();

        // Timing section
        report.push_str("=== Timing Profile ===\n");
        let total_time: Duration = self.timing_summaries.iter().map(|s| s.total).sum();

        for summary in &self.timing_summaries {
            report.push_str(&format!(
                "{}: {:.3}ms ({:.1}%) - {} calls, avg {:.3}ms\n",
                summary.name,
                summary.total.as_secs_f64() * 1000.0,
                summary.percentage_of(total_time),
                summary.count,
                summary.mean.as_secs_f64() * 1000.0
            ));
        }

        // Memory section
        report.push_str("\n=== Memory Profile ===\n");
        report.push_str(&format!(
            "Peak Usage: {:.2} MB\n\
             Current Usage: {:.2} MB\n\
             Allocations: {}\n\
             Deallocations: {}\n\
             Efficiency: {:.1}%\n\
             Fragmentation: {:.1}%\n",
            self.memory_profile.peak_usage as f64 / 1_048_576.0,
            self.memory_profile.current_usage as f64 / 1_048_576.0,
            self.memory_profile.total_allocations,
            self.memory_profile.total_deallocations,
            self.memory_profile.efficiency() * 100.0,
            self.memory_profile.fragmentation() * 100.0
        ));

        // Cache section
        report.push_str("\n=== Cache Profile ===\n");
        report.push_str(&format!(
            "L1 Hit Rate: {:.1}%\n\
             L2 Hit Rate: {:.1}%\n\
             L3 Hit Rate: {:.1}%\n\
             TLB Hit Rate: {:.1}%\n\
             Overall Efficiency: {:.1}%\n",
            self.cache_profile.statistics.l1_hit_rate() * 100.0,
            self.cache_profile.statistics.l2_hit_rate() * 100.0,
            self.cache_profile.statistics.l3_hit_rate() * 100.0,
            self.cache_profile.statistics.tlb_hit_rate() * 100.0,
            self.cache_profile.efficiency() * 100.0
        ));

        // Performance analysis
        report.push_str("\n=== Performance Analysis ===\n");
        report.push_str(&self.performance_analysis);

        report
    }
}
