use super::{EdgeRuntime, EdgeRuntimePerformanceMonitor};

impl EdgeRuntime {
    pub(super) fn update_performance_stats(&mut self, inference_time_us: u64) {
        self.performance_monitor.inference_count += 1;
        self.performance_monitor.total_inference_time_us += inference_time_us;

        self.performance_monitor.avg_latency_us = self.performance_monitor.total_inference_time_us
            as f64
            / self.performance_monitor.inference_count as f64;

        let current_memory = self.allocator.get_allocated_memory();
        if current_memory > self.performance_monitor.peak_memory_usage {
            self.performance_monitor.peak_memory_usage = current_memory;
        }

        self.performance_monitor.memory_efficiency =
            current_memory as f32 / self.allocator.total_memory as f32;
    }

    pub fn get_performance_stats(&self) -> &EdgeRuntimePerformanceMonitor {
        &self.performance_monitor
    }
}
