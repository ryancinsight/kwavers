//! GPU allocation statistics.

/// Statistics snapshot for GPU memory tracking.
#[derive(Debug, Default, Clone)]
pub struct GpuAllocationStats {
    /// Total number of allocations performed.
    pub total_allocations: usize,
    /// Total number of deallocations performed.
    pub total_deallocations: usize,
    /// Current memory usage in bytes.
    pub current_bytes: usize,
    /// Peak memory usage in bytes.
    pub peak_bytes: usize,
    /// Total memory budget in bytes.
    pub budget_bytes: usize,
    /// Estimated device memory capacity in bytes.
    pub device_capacity: usize,
}

impl GpuAllocationStats {
    /// Memory utilization ratio `current_bytes / device_capacity` (0.0 to 1.0).
    pub fn utilization(&self) -> f64 {
        if self.device_capacity == 0 {
            0.0
        } else {
            self.current_bytes as f64 / self.device_capacity as f64
        }
    }

    /// Peak utilization ratio `peak_bytes / device_capacity` (0.0 to 1.0).
    pub fn peak_utilization(&self) -> f64 {
        if self.device_capacity == 0 {
            0.0
        } else {
            self.peak_bytes as f64 / self.device_capacity as f64
        }
    }

    /// Available memory in bytes considering the given `safety_factor`.
    pub fn available_bytes(&self, safety_factor: f64) -> usize {
        let threshold = (self.device_capacity as f64 * safety_factor) as usize;
        threshold.saturating_sub(self.current_bytes)
    }

    /// Return `true` if allocating `size` bytes would exceed the budget.
    pub fn would_exceed_budget(&self, size: usize, safety_factor: f64) -> bool {
        let threshold = (self.device_capacity as f64 * safety_factor) as usize;
        self.current_bytes + size > threshold
    }
}
