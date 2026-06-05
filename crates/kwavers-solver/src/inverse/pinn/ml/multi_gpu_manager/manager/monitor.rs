//! Performance monitoring, metrics, and fault tolerance for multi-GPU manager.

use super::super::types::PerformanceSummary;
use super::MultiGpuManager;
use kwavers_core::error::{KwaversError, KwaversResult};

impl MultiGpuManager {
    /// Update performance metrics for a GPU.
    pub fn update_performance_metrics(
        &mut self,
        gpu_id: usize,
        utilization: f32,
        memory_used: usize,
    ) {
        if gpu_id < self.devices.len() {
            self.devices[gpu_id].compute_load = utilization;
            self.devices[gpu_id].memory_used = memory_used;

            self.performance_monitor.gpu_utilization[gpu_id].push(utilization);
            self.performance_monitor.memory_usage[gpu_id].push(memory_used);

            const MAX_HISTORY: usize = 100;
            if self.performance_monitor.gpu_utilization[gpu_id].len() > MAX_HISTORY {
                self.performance_monitor.gpu_utilization[gpu_id].remove(0);
                self.performance_monitor.memory_usage[gpu_id].remove(0);
            }
        }
    }

    /// Calculate the coefficient of variation of GPU loads (load imbalance metric).
    pub fn calculate_load_imbalance(&self) -> f32 {
        if self.devices.is_empty() {
            return 0.0;
        }
        let loads: Vec<f32> = self.devices.iter().map(|d| d.compute_load).collect();
        let mean_load = loads.iter().sum::<f32>() / loads.len() as f32;
        if mean_load == 0.0 {
            return 0.0;
        }
        let variance = loads
            .iter()
            .map(|&load| (load - mean_load).powi(2))
            .sum::<f32>()
            / loads.len() as f32;
        variance.sqrt() / mean_load
    }

    /// Calculate average GPU utilization as the scaling efficiency proxy.
    pub fn calculate_scaling_efficiency(&self) -> f64 {
        let n_gpus = self.devices.len() as f64;
        self.devices
            .iter()
            .map(|d| d.compute_load as f64)
            .sum::<f64>()
            / n_gpus
    }

    /// Get a consolidated performance summary.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn get_performance_summary(&self) -> PerformanceSummary {
        PerformanceSummary {
            num_gpus: self.devices.len(),
            load_imbalance: self.calculate_load_imbalance(),
            scaling_efficiency: self.calculate_scaling_efficiency(),
            communication_overhead: self.performance_monitor.communication_overhead,
            average_utilization: self.devices.iter().map(|d| d.compute_load).sum::<f32>()
                / self.devices.len() as f32,
        }
    }

    /// Handle a GPU failure by marking it unhealthy and redistributing its work.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn handle_gpu_failure(&mut self, failed_gpu_id: usize) -> KwaversResult<()> {
        if failed_gpu_id >= self.devices.len() {
            return Ok(());
        }
        self.devices[failed_gpu_id].healthy = false;

        if self.fault_tolerance.graceful_degradation {
            let failed_work: Vec<_> = self
                .active_work
                .values()
                .filter(|work| work.device_id == failed_gpu_id)
                .cloned()
                .collect();

            for work in failed_work {
                self.work_queue.push_back(work);
            }

            self.active_work
                .retain(|_, work| work.device_id != failed_gpu_id);

            self.reassign_work_after_failure()?;
        }

        Ok(())
    }

    fn reassign_work_after_failure(&mut self) -> KwaversResult<()> {
        let healthy_gpus: Vec<usize> = self
            .devices
            .iter()
            .enumerate()
            .filter(|(_, device)| device.healthy)
            .map(|(i, _)| i)
            .collect();

        if healthy_gpus.is_empty() {
            return Err(KwaversError::System(
                kwavers_core::error::SystemError::ResourceUnavailable {
                    resource: "No healthy GPUs remaining".to_string(),
                },
            ));
        }

        while let Some(work) = self.work_queue.pop_front() {
            let best_gpu = healthy_gpus[self.work_queue.len() % healthy_gpus.len()];
            let mut reassigned_work = work;
            reassigned_work.device_id = best_gpu;
            self.active_work.insert(reassigned_work.id, reassigned_work);
        }

        Ok(())
    }
}
