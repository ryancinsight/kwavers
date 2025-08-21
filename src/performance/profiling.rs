//! Performance profiling infrastructure
//!
//! This module provides comprehensive performance profiling capabilities
//! including timing, memory usage, cache behavior, and roofline analysis.
//!
//! ## Design Principles
//! - **Zero-Copy**: Profile data uses slices and iterators
//! - **Low Overhead**: Minimal impact on performance being measured
//! - **Composable**: Modular profiling components
//! - **Scientific**: Based on established performance models

use crate::grid::Grid;
use crate::KwaversResult;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Performance profiler for comprehensive analysis
#[derive(Debug)]
pub struct PerformanceProfiler {
    /// Timing measurements
    timings: Arc<Mutex<HashMap<String, Vec<Duration>>>>,
    /// Memory allocations
    memory_events: Arc<Mutex<Vec<MemoryEvent>>>,
    /// Cache statistics
    cache_stats: Arc<Mutex<CacheStatistics>>,
    /// Grid configuration for analysis
    grid: Grid,
}

/// Timing scope for RAII-based profiling
pub struct TimingScope {
    name: String,
    start: Instant,
    profiler: Arc<Mutex<HashMap<String, Vec<Duration>>>>,
}

/// Memory profiling event
#[derive(Debug, Clone)]
pub struct MemoryEvent {
    /// Event timestamp
    timestamp: Instant,
    /// Allocation size in bytes
    size: usize,
    /// Event type
    event_type: MemoryEventType,
    /// Optional description
    description: Option<String>,
}

/// Memory event types
#[derive(Debug, Clone, Copy)]
pub enum MemoryEventType {
    Allocation,
    Deallocation,
    Peak,
}

/// Cache profiling statistics
#[derive(Debug, Default)]
pub struct CacheStatistics {
    /// L1 cache hits
    l1_hits: u64,
    /// L1 cache misses
    l1_misses: u64,
    /// L2 cache hits
    l2_hits: u64,
    /// L2 cache misses
    l2_misses: u64,
    /// L3 cache hits
    l3_hits: u64,
    /// L3 cache misses
    l3_misses: u64,
}

/// Memory profile summary
#[derive(Debug)]
pub struct MemoryProfile {
    /// Peak memory usage
    peak_usage: usize,
    /// Current usage
    current_usage: usize,
    /// Total allocations
    total_allocations: usize,
    /// Allocation histogram by size
    size_histogram: HashMap<usize, usize>,
}

/// Cache profile summary
#[derive(Debug)]
pub struct CacheProfile {
    /// L1 hit rate
    l1_hit_rate: f64,
    /// L2 hit rate
    l2_hit_rate: f64,
    /// L3 hit rate
    l3_hit_rate: f64,
    /// Estimated memory bandwidth (GB/s)
    bandwidth_gbs: f64,
}

/// Roofline analysis results
#[derive(Debug)]
pub struct RooflineAnalysis {
    /// Arithmetic intensity (FLOPs/byte)
    arithmetic_intensity: f64,
    /// Achieved performance (GFLOP/s)
    achieved_gflops: f64,
    /// Peak performance (GFLOP/s)
    peak_gflops: f64,
    /// Memory bandwidth (GB/s)
    memory_bandwidth: f64,
    /// Performance bound type
    bound_type: PerformanceBound,
}

/// Performance bound classification
#[derive(Debug, Clone, Copy)]
pub enum PerformanceBound {
    /// Limited by memory bandwidth
    MemoryBound,
    /// Limited by compute capacity
    ComputeBound,
    /// Balanced (near roofline)
    Balanced,
}

/// Comprehensive profile report
#[derive(Debug)]
pub struct ProfileReport {
    /// Timing summary
    pub timings: HashMap<String, TimingSummary>,
    /// Memory profile
    pub memory: MemoryProfile,
    /// Cache profile
    pub cache: CacheProfile,
    /// Roofline analysis
    pub roofline: RooflineAnalysis,
    /// Grid updates per second
    pub grid_updates_per_second: f64,
}

/// Timing statistics summary
#[derive(Debug)]
pub struct TimingSummary {
    /// Total time spent
    pub total: Duration,
    /// Average time per call
    pub average: Duration,
    /// Minimum time
    pub min: Duration,
    /// Maximum time
    pub max: Duration,
    /// Number of calls
    pub count: usize,
    /// Standard deviation
    pub std_dev: Duration,
}

impl PerformanceProfiler {
    /// Create a new performance profiler
    pub fn new(grid: &Grid) -> Self {
        Self {
            timings: Arc::new(Mutex::new(HashMap::new())),
            memory_events: Arc::new(Mutex::new(Vec::new())),
            cache_stats: Arc::new(Mutex::new(CacheStatistics::default())),
            grid: grid.clone(),
        }
    }

    /// Start a timing scope
    pub fn time_scope(&self, name: &str) -> TimingScope {
        TimingScope {
            name: name.to_string(),
            start: Instant::now(),
            profiler: self.timings.clone(),
        }
    }

    /// Record a memory event
    pub fn record_memory_event(
        &self,
        size: usize,
        event_type: MemoryEventType,
        description: Option<String>,
    ) {
        let event = MemoryEvent {
            timestamp: Instant::now(),
            size,
            event_type,
            description,
        };

        if let Ok(mut events) = self.memory_events.lock() {
            events.push(event);
        }
    }

    /// Update cache statistics
    pub fn update_cache_stats<F>(&self, update_fn: F)
    where
        F: FnOnce(&mut CacheStatistics),
    {
        if let Ok(mut stats) = self.cache_stats.lock() {
            update_fn(&mut *stats);
        }
    }

    /// Generate comprehensive profile report
    pub fn generate_report(&self) -> KwaversResult<ProfileReport> {
        let timings = self.analyze_timings()?;
        let memory = self.analyze_memory()?;
        let cache = self.analyze_cache()?;
        let roofline = self.perform_roofline_analysis(&timings, &cache)?;
        let grid_updates_per_second = self.calculate_grid_updates_per_second(&timings)?;

        Ok(ProfileReport {
            timings,
            memory,
            cache,
            roofline,
            grid_updates_per_second,
        })
    }

    /// Analyze timing measurements
    fn analyze_timings(&self) -> KwaversResult<HashMap<String, TimingSummary>> {
        let timings = self.timings.lock().unwrap();

        timings
            .iter()
            .map(|(name, durations)| {
                let summary = self.calculate_timing_summary(durations);
                Ok((name.clone(), summary))
            })
            .collect()
    }

    /// Calculate timing statistics
    fn calculate_timing_summary(&self, durations: &[Duration]) -> TimingSummary {
        if durations.is_empty() {
            return TimingSummary {
                total: Duration::ZERO,
                average: Duration::ZERO,
                min: Duration::ZERO,
                max: Duration::ZERO,
                count: 0,
                std_dev: Duration::ZERO,
            };
        }

        let total: Duration = durations.iter().sum();
        let count = durations.len();
        let average = total / count as u32;

        let min = durations.iter().min().copied().unwrap_or(Duration::ZERO);
        let max = durations.iter().max().copied().unwrap_or(Duration::ZERO);

        // Calculate standard deviation using iterator
        let avg_nanos = average.as_nanos() as f64;
        let variance = durations
            .iter()
            .map(|d| {
                let diff = d.as_nanos() as f64 - avg_nanos;
                diff * diff
            })
            .sum::<f64>()
            / count as f64;

        let std_dev = Duration::from_nanos(variance.sqrt() as u64);

        TimingSummary {
            total,
            average,
            min,
            max,
            count,
            std_dev,
        }
    }

    /// Analyze memory usage
    fn analyze_memory(&self) -> KwaversResult<MemoryProfile> {
        let events = self.memory_events.lock().unwrap();

        let mut current_usage = 0usize;
        let mut peak_usage = 0usize;
        let mut total_allocations = 0usize;
        let mut size_histogram = HashMap::new();

        // Process events using iterator
        events.iter().for_each(|event| {
            match event.event_type {
                MemoryEventType::Allocation => {
                    current_usage += event.size;
                    peak_usage = peak_usage.max(current_usage);
                    total_allocations += 1;

                    // Update histogram with power-of-2 buckets
                    let bucket = event.size.next_power_of_two();
                    *size_histogram.entry(bucket).or_insert(0) += 1;
                }
                MemoryEventType::Deallocation => {
                    current_usage = current_usage.saturating_sub(event.size);
                }
                MemoryEventType::Peak => {
                    peak_usage = peak_usage.max(event.size);
                }
            }
        });

        Ok(MemoryProfile {
            peak_usage,
            current_usage,
            total_allocations,
            size_histogram,
        })
    }

    /// Analyze cache behavior
    fn analyze_cache(&self) -> KwaversResult<CacheProfile> {
        let stats = self.cache_stats.lock().unwrap();

        let l1_hit_rate = if stats.l1_hits + stats.l1_misses > 0 {
            stats.l1_hits as f64 / (stats.l1_hits + stats.l1_misses) as f64
        } else {
            0.0
        };

        let l2_hit_rate = if stats.l2_hits + stats.l2_misses > 0 {
            stats.l2_hits as f64 / (stats.l2_hits + stats.l2_misses) as f64
        } else {
            0.0
        };

        let l3_hit_rate = if stats.l3_hits + stats.l3_misses > 0 {
            stats.l3_hits as f64 / (stats.l3_hits + stats.l3_misses) as f64
        } else {
            0.0
        };

        // Estimate bandwidth based on cache misses and typical cache line size
        let cache_line_size = 64; // bytes
        let total_misses = stats.l1_misses + stats.l2_misses + stats.l3_misses;
        let bytes_transferred = total_misses * cache_line_size;

        // Default 1 second runtime for bandwidth calculation (adjusted with actual timing)
        let bandwidth_gbs = bytes_transferred as f64 / 1e9;

        Ok(CacheProfile {
            l1_hit_rate,
            l2_hit_rate,
            l3_hit_rate,
            bandwidth_gbs,
        })
    }

    /// Perform roofline analysis
    fn perform_roofline_analysis(
        &self,
        timings: &HashMap<String, TimingSummary>,
        cache: &CacheProfile,
    ) -> KwaversResult<RooflineAnalysis> {
        // Calculate arithmetic intensity for acoustic wave propagation
        // Typical stencil: 7 points, 2 arrays (pressure + velocity)
        let flops_per_point = 14.0; // 7 adds + 7 muls
        let bytes_per_point = 16.0; // 2 doubles read/write
        let arithmetic_intensity = flops_per_point / bytes_per_point;

        // Calculate achieved performance
        let total_points = self.grid.nx * self.grid.ny * self.grid.nz;
        let total_flops = total_points as f64 * flops_per_point;

        let compute_time = timings
            .values()
            .filter(|t| t.count > 0)
            .map(|t| t.total.as_secs_f64())
            .sum::<f64>();

        let achieved_gflops = if compute_time > 0.0 {
            total_flops / compute_time / 1e9
        } else {
            0.0
        };

        // Typical peak performance for modern CPUs
        let peak_gflops = 100.0; // Adjust based on actual hardware

        // Determine performance bound
        let roofline_compute = peak_gflops;
        let roofline_memory = cache.bandwidth_gbs * arithmetic_intensity;

        let bound_type = if achieved_gflops < 0.9 * roofline_memory.min(roofline_compute) {
            if roofline_memory < roofline_compute {
                PerformanceBound::MemoryBound
            } else {
                PerformanceBound::ComputeBound
            }
        } else {
            PerformanceBound::Balanced
        };

        Ok(RooflineAnalysis {
            arithmetic_intensity,
            achieved_gflops,
            peak_gflops,
            memory_bandwidth: cache.bandwidth_gbs,
            bound_type,
        })
    }

    /// Calculate grid updates per second
    fn calculate_grid_updates_per_second(
        &self,
        timings: &HashMap<String, TimingSummary>,
    ) -> KwaversResult<f64> {
        let total_points = self.grid.nx * self.grid.ny * self.grid.nz;

        // Find main computation timing
        let compute_time = timings
            .iter()
            .filter(|(name, _)| name.contains("compute") || name.contains("update"))
            .map(|(_, summary)| summary.total.as_secs_f64())
            .sum::<f64>();

        if compute_time > 0.0 {
            Ok(total_points as f64 / compute_time)
        } else {
            Ok(0.0)
        }
    }
}

impl Drop for TimingScope {
    fn drop(&mut self) {
        let duration = self.start.elapsed();
        if let Ok(mut timings) = self.profiler.lock() {
            timings
                .entry(self.name.clone())
                .or_insert_with(Vec::new)
                .push(duration);
        }
    }
}

impl ProfileReport {
    /// Print a formatted report
    pub fn print_summary(&self) {
        println!("\n=== Performance Profile Report ===\n");

        println!(
            "Grid Updates/Second: {:.2}M",
            self.grid_updates_per_second / 1e6
        );

        println!("\nTiming Summary:");
        let mut timing_entries: Vec<_> = self.timings.iter().collect();
        timing_entries.sort_by_key(|(_, summary)| std::cmp::Reverse(summary.total));

        for (name, summary) in timing_entries.iter().take(10) {
            println!(
                "  {}: {:.2}ms total ({} calls, {:.2}μs avg)",
                name,
                summary.total.as_secs_f64() * 1000.0,
                summary.count,
                summary.average.as_secs_f64() * 1e6
            );
        }

        println!("\nMemory Profile:");
        println!(
            "  Peak Usage: {:.2} MB",
            self.memory.peak_usage as f64 / 1e6
        );
        println!(
            "  Current Usage: {:.2} MB",
            self.memory.current_usage as f64 / 1e6
        );
        println!("  Total Allocations: {}", self.memory.total_allocations);

        println!("\nCache Profile:");
        println!("  L1 Hit Rate: {:.1}%", self.cache.l1_hit_rate * 100.0);
        println!("  L2 Hit Rate: {:.1}%", self.cache.l2_hit_rate * 100.0);
        println!("  L3 Hit Rate: {:.1}%", self.cache.l3_hit_rate * 100.0);
        println!("  Bandwidth: {:.1} GB/s", self.cache.bandwidth_gbs);

        println!("\nRoofline Analysis:");
        println!(
            "  Arithmetic Intensity: {:.2} FLOP/byte",
            self.roofline.arithmetic_intensity
        );
        println!("  Achieved: {:.1} GFLOP/s", self.roofline.achieved_gflops);
        println!("  Peak: {:.1} GFLOP/s", self.roofline.peak_gflops);
        println!("  Performance Bound: {:?}", self.roofline.bound_type);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_profiler() {
        let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3);
        let profiler = PerformanceProfiler::new(&grid);

        // Test timing scope
        {
            let _scope = profiler.time_scope("test_operation");
            std::thread::sleep(Duration::from_millis(10));
        }

        // Test memory event
        profiler.record_memory_event(
            1024 * 1024,
            MemoryEventType::Allocation,
            Some("test allocation".to_string()),
        );

        // Test cache stats
        profiler.update_cache_stats(|stats| {
            stats.l1_hits = 1000;
            stats.l1_misses = 100;
        });

        // Generate report
        let report = profiler.generate_report().unwrap();
        assert!(report.timings.contains_key("test_operation"));
        assert_eq!(report.memory.total_allocations, 1);
        assert!(report.cache.l1_hit_rate > 0.9);
    }

    #[test]
    fn test_timing_summary_calculation() {
        let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3);
        let profiler = PerformanceProfiler::new(&grid);

        let durations = vec![
            Duration::from_millis(10),
            Duration::from_millis(12),
            Duration::from_millis(8),
            Duration::from_millis(11),
            Duration::from_millis(9),
        ];

        let summary = profiler.calculate_timing_summary(&durations);
        assert_eq!(summary.count, 5);
        assert_eq!(summary.min, Duration::from_millis(8));
        assert_eq!(summary.max, Duration::from_millis(12));
        assert_eq!(summary.total, Duration::from_millis(50));
    }
}
