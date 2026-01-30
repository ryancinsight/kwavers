//! GPU Real-Time Performance Monitoring
//!
//! Tracks execution metrics for GPU-accelerated multiphysics simulations,
//! enforces real-time budgets, and identifies performance bottlenecks.

use std::collections::HashMap;
use std::collections::VecDeque;

/// Performance metrics for real-time simulation
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Average step execution time (milliseconds)
    pub avg_step_time_ms: f64,

    /// 95th percentile step time (milliseconds)
    pub p95_step_time_ms: f64,

    /// 99th percentile step time (milliseconds)
    pub p99_step_time_ms: f64,

    /// GPU utilization percentage (0-100)
    pub gpu_utilization: f64,

    /// Data transfer overhead percentage
    pub transfer_overhead_percent: f64,

    /// I/O operation overhead percentage
    pub io_overhead_percent: f64,

    /// Percentage of steps within budget
    pub budget_satisfaction: f64,

    /// Estimated throughput (steps per second)
    pub throughput_steps_per_sec: f64,
}

/// Bottleneck analysis for real-time performance
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BottleneckType {
    /// GPU compute is the limiting factor
    GPUCompute,

    /// GPU-CPU data transfer is limiting
    DataTransfer,

    /// I/O operations are limiting
    IO,

    /// CPU preprocessing is limiting
    CPUPreprocessing,

    /// Within budget, no bottleneck
    None,
}

/// Budget analysis result
#[derive(Debug, Clone)]
pub struct BudgetAnalysis {
    /// Whether currently within budget
    pub within_budget: bool,

    /// Percentage over budget (0 if within)
    pub overage_percent: f64,

    /// Identified bottleneck
    pub bottleneck: BottleneckType,

    /// Recommendation for improvement
    pub recommendation: String,
}

/// Real-time performance monitor
#[derive(Debug)]
pub struct PerformanceMonitor {
    /// Step execution times (milliseconds)
    step_times: VecDeque<f64>,

    /// Kernel execution times per type
    kernel_times: HashMap<String, VecDeque<f64>>,

    /// Data transfer times (milliseconds)
    transfer_times: VecDeque<f64>,

    /// I/O operation times (milliseconds)
    io_times: VecDeque<f64>,

    /// Real-time budget (milliseconds)
    budget_ms: f64,

    /// History window size
    window_size: usize,

    /// Total steps recorded
    total_steps: u64,

    /// Steps exceeding budget
    budget_violations: u64,
}

impl PerformanceMonitor {
    /// Create new performance monitor
    pub fn new(budget_ms: f64, window_size: usize) -> Self {
        Self {
            step_times: VecDeque::with_capacity(window_size),
            kernel_times: HashMap::new(),
            transfer_times: VecDeque::with_capacity(window_size),
            io_times: VecDeque::with_capacity(window_size),
            budget_ms,
            window_size,
            total_steps: 0,
            budget_violations: 0,
        }
    }

    /// Record a step execution time
    pub fn record_step(&mut self, time_ms: f64) {
        self.step_times.push_back(time_ms);
        if self.step_times.len() > self.window_size {
            self.step_times.pop_front();
        }

        self.total_steps += 1;
        if time_ms > self.budget_ms {
            self.budget_violations += 1;
        }
    }

    /// Record kernel execution time
    pub fn record_kernel(&mut self, name: String, time_ms: f64) {
        self.kernel_times
            .entry(name.clone())
            .or_insert_with(|| VecDeque::with_capacity(self.window_size))
            .push_back(time_ms);

        // Trim to window size
        if let Some(times) = self.kernel_times.get_mut(&name) {
            if times.len() > self.window_size {
                times.pop_front();
            }
        }
    }

    /// Record data transfer time
    pub fn record_transfer(&mut self, time_ms: f64) {
        self.transfer_times.push_back(time_ms);
        if self.transfer_times.len() > self.window_size {
            self.transfer_times.pop_front();
        }
    }

    /// Record I/O operation time
    pub fn record_io(&mut self, time_ms: f64) {
        self.io_times.push_back(time_ms);
        if self.io_times.len() > self.window_size {
            self.io_times.pop_front();
        }
    }

    /// Get current performance metrics
    pub fn get_metrics(&self) -> PerformanceMetrics {
        let avg_step = self.calculate_average(&self.step_times);
        let p95_step = self.calculate_percentile(&self.step_times, 0.95);
        let p99_step = self.calculate_percentile(&self.step_times, 0.99);

        let avg_transfer = self.calculate_average(&self.transfer_times);
        let avg_io = self.calculate_average(&self.io_times);
        let _total_overhead = avg_transfer + avg_io;

        let transfer_overhead_percent = if avg_step > 0.0 {
            (avg_transfer / avg_step) * 100.0
        } else {
            0.0
        };

        let io_overhead_percent = if avg_step > 0.0 {
            (avg_io / avg_step) * 100.0
        } else {
            0.0
        };

        let budget_satisfaction = if self.total_steps > 0 {
            ((self.total_steps - self.budget_violations) as f64 / self.total_steps as f64) * 100.0
        } else {
            100.0
        };

        let throughput_steps_per_sec = if avg_step > 0.0 {
            1000.0 / avg_step
        } else {
            0.0
        };

        PerformanceMetrics {
            avg_step_time_ms: avg_step,
            p95_step_time_ms: p95_step,
            p99_step_time_ms: p99_step,
            gpu_utilization: 100.0 - transfer_overhead_percent,
            transfer_overhead_percent,
            io_overhead_percent,
            budget_satisfaction,
            throughput_steps_per_sec,
        }
    }

    /// Analyze budget status
    pub fn analyze_budget(&self) -> BudgetAnalysis {
        let metrics = self.get_metrics();

        let (within_budget, overage) = if metrics.avg_step_time_ms <= self.budget_ms {
            (true, 0.0)
        } else {
            let overage_percent =
                ((metrics.avg_step_time_ms - self.budget_ms) / self.budget_ms) * 100.0;
            (false, overage_percent)
        };

        let bottleneck = if metrics.gpu_utilization < 70.0 {
            BottleneckType::CPUPreprocessing
        } else if metrics.transfer_overhead_percent > metrics.io_overhead_percent {
            BottleneckType::DataTransfer
        } else if metrics.io_overhead_percent > 5.0 {
            BottleneckType::IO
        } else {
            BottleneckType::GPUCompute
        };

        let recommendation = match bottleneck {
            BottleneckType::GPUCompute => {
                "GPU is fully utilized. Consider: larger grid, more physics, or accept throughput."
                    .to_string()
            }
            BottleneckType::DataTransfer => {
                "Optimize GPU-CPU transfers: use pinned memory, async transfers, or Phase 1 interpolation."
                    .to_string()
            }
            BottleneckType::IO => {
                "I/O is bottleneck: reduce checkpoint frequency or enable async I/O."
                    .to_string()
            }
            BottleneckType::CPUPreprocessing => {
                "CPU preprocessing is slow: profile and optimize boundary conditions, CFL computation."
                    .to_string()
            }
            BottleneckType::None => "Within budget. Increase workload to utilize resources better.".to_string(),
        };

        BudgetAnalysis {
            within_budget,
            overage_percent: overage,
            bottleneck,
            recommendation,
        }
    }

    /// Check if currently within budget
    pub fn is_within_budget(&self) -> bool {
        if let Some(&last_time) = self.step_times.back() {
            last_time <= self.budget_ms
        } else {
            true
        }
    }

    /// Get estimated remaining budget for next step
    pub fn estimated_remaining_budget(&self) -> f64 {
        let avg_time = self.calculate_average(&self.step_times);
        (self.budget_ms - avg_time).max(0.0)
    }

    // ========== Private Methods ==========

    fn calculate_average(&self, values: &VecDeque<f64>) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        values.iter().sum::<f64>() / values.len() as f64
    }

    fn calculate_percentile(&self, values: &VecDeque<f64>, percentile: f64) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let mut sorted: Vec<f64> = values.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let index = ((percentile * (sorted.len() as f64)) as usize).min(sorted.len() - 1);
        sorted[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monitor_creation() {
        let monitor = PerformanceMonitor::new(10.0, 100);
        assert_eq!(monitor.budget_ms, 10.0);
        assert_eq!(monitor.total_steps, 0);
    }

    #[test]
    fn test_step_recording() {
        let mut monitor = PerformanceMonitor::new(10.0, 10);

        monitor.record_step(5.0);
        monitor.record_step(8.0);
        monitor.record_step(6.0);

        assert_eq!(monitor.total_steps, 3);
        assert_eq!(monitor.budget_violations, 0);

        let metrics = monitor.get_metrics();
        assert!((metrics.avg_step_time_ms - 6.333).abs() < 0.01);
    }

    #[test]
    fn test_budget_violation_detection() {
        let mut monitor = PerformanceMonitor::new(10.0, 10);

        monitor.record_step(8.0);
        monitor.record_step(12.0); // Exceeds budget
        monitor.record_step(9.0);

        assert_eq!(monitor.budget_violations, 1);
        let metrics = monitor.get_metrics();
        assert!(metrics.budget_satisfaction < 100.0);
    }

    #[test]
    fn test_bottleneck_detection() {
        let mut monitor = PerformanceMonitor::new(10.0, 10);

        // Simulate high transfer overhead
        for _ in 0..5 {
            monitor.record_step(10.0);
            monitor.record_transfer(6.0);
        }

        let analysis = monitor.analyze_budget();
        assert_eq!(analysis.bottleneck, BottleneckType::DataTransfer);
    }

    #[test]
    fn test_percentile_calculation() {
        let mut monitor = PerformanceMonitor::new(100.0, 100);

        for i in 1..=100 {
            monitor.record_step(i as f64);
        }

        let metrics = monitor.get_metrics();
        assert!((metrics.p95_step_time_ms - 95.0).abs() < 2.0);
        assert!((metrics.p99_step_time_ms - 99.0).abs() < 2.0);
    }
}
