//! Load balancing and work assignment for multi-GPU manager.

use super::super::types::{LoadBalancingAlgorithm, WorkUnit};
use super::MultiGpuManager;
use kwavers_core::error::KwaversResult;

impl MultiGpuManager {
    /// Assign work to GPUs using the configured load balancing algorithm.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn assign_work(&mut self, work_units: Vec<WorkUnit>) -> KwaversResult<()> {
        match &self.load_balancer {
            LoadBalancingAlgorithm::Static => self.assign_work_static(work_units),
            LoadBalancingAlgorithm::Dynamic { .. } => self.assign_work_dynamic(work_units),
            LoadBalancingAlgorithm::Predictive { .. } => self.assign_work_predictive(work_units),
        }
    }

    fn assign_work_static(&mut self, work_units: Vec<WorkUnit>) -> KwaversResult<()> {
        let n_gpus = self.devices.len();
        for (i, work_unit) in work_units.into_iter().enumerate() {
            let gpu_id = i % n_gpus;
            let mut assigned_work = work_unit;
            assigned_work.device_id = gpu_id;
            self.active_work.insert(assigned_work.id, assigned_work);
        }
        Ok(())
    }

    fn assign_work_dynamic(&mut self, work_units: Vec<WorkUnit>) -> KwaversResult<()> {
        let mut sorted_work = work_units;
        sorted_work.sort_by(|a, b| b.complexity.total_cmp(&a.complexity));
        for work_unit in sorted_work {
            let best_gpu = self.find_least_loaded_gpu();
            let mut assigned_work = work_unit;
            assigned_work.device_id = best_gpu;
            self.active_work.insert(assigned_work.id, assigned_work);
        }
        Ok(())
    }

    fn assign_work_predictive(&mut self, work_units: Vec<WorkUnit>) -> KwaversResult<()> {
        for work_unit in work_units {
            let predicted_gpu = self.predict_optimal_gpu(&work_unit);
            let mut assigned_work = work_unit;
            assigned_work.device_id = predicted_gpu;
            self.active_work.insert(assigned_work.id, assigned_work);
        }
        Ok(())
    }

    pub(super) fn find_least_loaded_gpu(&self) -> usize {
        let mut min_load = f32::INFINITY;
        let mut best_gpu = 0;
        for (i, device) in self.devices.iter().enumerate() {
            if device.healthy && device.compute_load < min_load {
                min_load = device.compute_load;
                best_gpu = i;
            }
        }
        best_gpu
    }

    fn predict_optimal_gpu(&self, _work_unit: &WorkUnit) -> usize {
        self.find_least_loaded_gpu()
    }
}
