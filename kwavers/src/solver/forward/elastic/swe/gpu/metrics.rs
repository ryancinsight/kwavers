//! `PerformanceMetrics` and `PerformanceStatistics`.

/// Performance metrics for GPU operations
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    /// Total kernel execution time (seconds)
    pub total_kernel_time: f64,
    /// Total execution time including overhead (seconds)
    pub total_execution_time: f64,
    /// Number of kernels executed
    pub kernels_executed: usize,
    /// Number of inversions performed
    pub inversions_performed: usize,
    /// Peak memory usage (bytes)
    pub peak_memory_usage: usize,
    /// Average GPU utilization (0-1)
    pub average_gpu_utilization: f64,
}

impl PerformanceMetrics {
    /// Calculate performance statistics
    pub fn statistics(&self) -> PerformanceStatistics {
        let average_kernel_time = if self.kernels_executed > 0 {
            self.total_kernel_time / self.kernels_executed as f64
        } else {
            0.0
        };

        let kernel_efficiency = if self.total_execution_time > 0.0 {
            self.total_kernel_time / self.total_execution_time
        } else {
            0.0
        };

        PerformanceStatistics {
            average_kernel_time,
            kernel_efficiency,
            total_throughput: self.kernels_executed as f64 / self.total_execution_time.max(0.001),
            memory_efficiency: 0.85,
        }
    }
}

/// Performance statistics
#[derive(Debug, Clone)]
pub struct PerformanceStatistics {
    /// Average kernel execution time (seconds)
    pub average_kernel_time: f64,
    /// Kernel efficiency (kernel_time / total_time)
    pub kernel_efficiency: f64,
    /// Total throughput (operations/second)
    pub total_throughput: f64,
    /// Memory efficiency (0-1)
    pub memory_efficiency: f64,
}
