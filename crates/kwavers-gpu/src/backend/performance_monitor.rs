//! GPU Real-Time Performance Monitoring
//!
//! Tracks execution metrics for GPU-accelerated multiphysics simulations,
//! enforces real-time budgets, and identifies performance bottlenecks.

use std::collections::HashMap;
use std::collections::VecDeque;

use aequitas::systems::si::quantities::{Dimensionless, Frequency, Time};

/// Performance metrics for real-time simulation
#[derive(Debug, Clone)]
pub struct GpuStepMetrics {
    /// Average step execution time.
    pub avg_step_time: Time<f64>,

    /// 95th percentile step time.
    pub p95_step_time: Time<f64>,

    /// 99th percentile step time.
    pub p99_step_time: Time<f64>,

    /// GPU utilization percentage (0-100), represented as dimensionless data.
    pub gpu_utilization: Dimensionless<f64>,

    /// Data transfer overhead percentage, represented as dimensionless data.
    pub transfer_overhead: Dimensionless<f64>,

    /// I/O operation overhead percentage, represented as dimensionless data.
    pub io_overhead: Dimensionless<f64>,

    /// Percentage of steps within budget, represented as dimensionless data.
    pub budget_satisfaction: Dimensionless<f64>,

    /// Estimated throughput in steps per second.
    pub throughput: Frequency<f64>,
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

    /// Percentage over budget (0 if within), represented as dimensionless data.
    pub overage: Dimensionless<f64>,

    /// Identified bottleneck
    pub bottleneck: BottleneckType,

    /// Recommendation for improvement
    pub recommendation: String,
}

/// Real-time performance monitor
#[derive(Debug)]
pub struct GpuPerformanceMonitor {
    /// Step execution times.
    step_times: VecDeque<Time<f64>>,

    /// Kernel execution times per type
    kernel_times: HashMap<String, VecDeque<Time<f64>>>,

    /// Data transfer times.
    transfer_times: VecDeque<Time<f64>>,

    /// I/O operation times.
    io_times: VecDeque<Time<f64>>,

    /// Real-time budget.
    budget: Time<f64>,

    /// History window size
    window_size: usize,

    /// Total steps recorded
    total_steps: u64,

    /// Steps exceeding budget
    budget_violations: u64,
}

impl GpuPerformanceMonitor {
    /// Create new performance monitor
    pub fn new(budget: Time<f64>, window_size: usize) -> Self {
        Self {
            step_times: VecDeque::with_capacity(window_size),
            kernel_times: HashMap::new(),
            transfer_times: VecDeque::with_capacity(window_size),
            io_times: VecDeque::with_capacity(window_size),
            budget,
            window_size,
            total_steps: 0,
            budget_violations: 0,
        }
    }

    /// Record a step execution time
    pub fn record_step(&mut self, time: Time<f64>) {
        self.step_times.push_back(time);
        if self.step_times.len() > self.window_size {
            self.step_times.pop_front();
        }

        self.total_steps += 1;
        if time.into_base() > self.budget.into_base() {
            self.budget_violations += 1;
        }
    }

    /// Record kernel execution time
    pub fn record_kernel(&mut self, name: String, time: Time<f64>) {
        self.kernel_times
            .entry(name.clone())
            .or_insert_with(|| VecDeque::with_capacity(self.window_size))
            .push_back(time);

        // Trim to window size
        if let Some(times) = self.kernel_times.get_mut(&name) {
            if times.len() > self.window_size {
                times.pop_front();
            }
        }
    }

    /// Record data transfer time
    pub fn record_transfer(&mut self, time: Time<f64>) {
        self.transfer_times.push_back(time);
        if self.transfer_times.len() > self.window_size {
            self.transfer_times.pop_front();
        }
    }

    /// Record I/O operation time
    pub fn record_io(&mut self, time: Time<f64>) {
        self.io_times.push_back(time);
        if self.io_times.len() > self.window_size {
            self.io_times.pop_front();
        }
    }

    /// Get current performance metrics
    pub fn get_metrics(&self) -> GpuStepMetrics {
        let avg_step = self.calculate_average(&self.step_times);
        let p95_step = self.calculate_percentile(&self.step_times, 0.95);
        let p99_step = self.calculate_percentile(&self.step_times, 0.99);

        let avg_transfer = self.calculate_average(&self.transfer_times);
        let avg_io = self.calculate_average(&self.io_times);

        let transfer_overhead = if avg_step.into_base() > 0.0 {
            Dimensionless::from_base((avg_transfer.into_base() / avg_step.into_base()) * 100.0)
        } else {
            Dimensionless::from_base(0.0)
        };

        let io_overhead = if avg_step.into_base() > 0.0 {
            Dimensionless::from_base((avg_io.into_base() / avg_step.into_base()) * 100.0)
        } else {
            Dimensionless::from_base(0.0)
        };

        let budget_satisfaction = if self.total_steps > 0 {
            Dimensionless::from_base(
                ((self.total_steps - self.budget_violations) as f64 / self.total_steps as f64)
                    * 100.0,
            )
        } else {
            Dimensionless::from_base(100.0)
        };

        let throughput = if avg_step.into_base() > 0.0 {
            Frequency::from_base(1.0 / avg_step.into_base())
        } else {
            Frequency::from_base(0.0)
        };

        GpuStepMetrics {
            avg_step_time: avg_step,
            p95_step_time: p95_step,
            p99_step_time: p99_step,
            gpu_utilization: Dimensionless::from_base(100.0 - transfer_overhead.into_base()),
            transfer_overhead,
            io_overhead,
            budget_satisfaction,
            throughput,
        }
    }

    /// Analyze budget status
    pub fn analyze_budget(&self) -> BudgetAnalysis {
        let metrics = self.get_metrics();

        let (within_budget, overage) =
            if metrics.avg_step_time.into_base() <= self.budget.into_base() {
                (true, Dimensionless::from_base(0.0))
            } else {
                let overage = Dimensionless::from_base(
                    ((metrics.avg_step_time.into_base() - self.budget.into_base())
                        / self.budget.into_base())
                        * 100.0,
                );
                (false, overage)
            };

        let bottleneck = if metrics.transfer_overhead.into_base() > metrics.io_overhead.into_base()
        {
            BottleneckType::DataTransfer
        } else if metrics.io_overhead.into_base() > 5.0 {
            BottleneckType::IO
        } else if metrics.gpu_utilization.into_base() < 70.0 {
            BottleneckType::CPUPreprocessing
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
            overage,
            bottleneck,
            recommendation,
        }
    }

    /// Check if currently within budget
    pub fn is_within_budget(&self) -> bool {
        if let Some(&last_time) = self.step_times.back() {
            last_time.into_base() <= self.budget.into_base()
        } else {
            true
        }
    }

    /// Number of recorded steps that exceeded the configured budget.
    #[must_use]
    pub fn budget_violations(&self) -> u64 {
        self.budget_violations
    }

    /// Get estimated remaining budget for next step
    #[must_use]
    pub fn estimated_remaining_budget(&self) -> Time<f64> {
        let avg_time = self.calculate_average(&self.step_times);
        Time::from_base((self.budget.into_base() - avg_time.into_base()).max(0.0))
    }

    // ========== Private Methods ==========

    fn calculate_average(&self, values: &VecDeque<Time<f64>>) -> Time<f64> {
        if values.is_empty() {
            return Time::from_base(0.0);
        }
        Time::from_base(
            values.iter().map(|value| value.into_base()).sum::<f64>() / values.len() as f64,
        )
    }

    fn calculate_percentile(&self, values: &VecDeque<Time<f64>>, percentile: f64) -> Time<f64> {
        if values.is_empty() {
            return Time::from_base(0.0);
        }

        let mut sorted: Vec<f64> = values.iter().map(|value| value.into_base()).collect();
        sorted.sort_by(|a, b| a.total_cmp(b));

        let index = ((percentile * (sorted.len() as f64)) as usize).min(sorted.len() - 1);
        Time::from_base(sorted[index])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aequitas::systems::si::units::Millisecond;

    fn milliseconds(value: f64) -> Time<f64> {
        Time::from_unit::<Millisecond>(value)
    }

    #[test]
    fn test_monitor_creation() {
        let monitor = GpuPerformanceMonitor::new(milliseconds(10.0), 100);
        assert_eq!(monitor.budget, milliseconds(10.0));
        assert_eq!(monitor.total_steps, 0);
    }

    #[test]
    fn test_step_recording() {
        let mut monitor = GpuPerformanceMonitor::new(milliseconds(10.0), 10);

        monitor.record_step(milliseconds(5.0));
        monitor.record_step(milliseconds(8.0));
        monitor.record_step(milliseconds(6.0));

        assert_eq!(monitor.total_steps, 3);
        assert_eq!(monitor.budget_violations, 0);

        let metrics = monitor.get_metrics();
        assert!((metrics.avg_step_time.in_unit::<Millisecond>() - 6.333).abs() < 0.01);
    }

    #[test]
    fn test_budget_violation_detection() {
        let mut monitor = GpuPerformanceMonitor::new(milliseconds(10.0), 10);

        monitor.record_step(milliseconds(8.0));
        monitor.record_step(milliseconds(12.0)); // Exceeds budget
        monitor.record_step(milliseconds(9.0));

        assert_eq!(monitor.budget_violations, 1);
        let metrics = monitor.get_metrics();
        assert!(metrics.budget_satisfaction.into_base() < 100.0);
    }

    #[test]
    fn test_bottleneck_detection() {
        let mut monitor = GpuPerformanceMonitor::new(milliseconds(10.0), 10);

        // Simulate high transfer overhead
        for _ in 0..5 {
            monitor.record_step(milliseconds(10.0));
            monitor.record_transfer(milliseconds(6.0));
        }

        let analysis = monitor.analyze_budget();
        assert_eq!(analysis.bottleneck, BottleneckType::DataTransfer);
    }

    #[test]
    fn test_percentile_calculation() {
        let mut monitor = GpuPerformanceMonitor::new(milliseconds(100.0), 100);

        for i in 1..=100 {
            monitor.record_step(milliseconds(i as f64));
        }

        let metrics = monitor.get_metrics();
        assert!((metrics.p95_step_time.in_unit::<Millisecond>() - 95.0).abs() < 2.0);
        assert!((metrics.p99_step_time.in_unit::<Millisecond>() - 99.0).abs() < 2.0);
    }
}
