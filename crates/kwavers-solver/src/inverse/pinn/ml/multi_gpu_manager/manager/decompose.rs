//! Decomposition strategies for multi-GPU task distribution.

use super::super::types::{MultiGpuDecompositionStrategy, WorkUnit};
use super::MultiGpuManager;
use kwavers_core::error::KwaversResult;

impl MultiGpuManager {
    /// Decompose a PINN training task across GPUs.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn decompose_training_task(
        &self,
        total_collocation_points: usize,
        geometry_bounds: (f64, f64, f64, f64),
    ) -> KwaversResult<Vec<WorkUnit>> {
        match &self.decomposition {
            MultiGpuDecompositionStrategy::Spatial {
                dimensions,
                overlap,
            } => self.decompose_spatially(
                total_collocation_points,
                geometry_bounds,
                *dimensions,
                *overlap,
            ),
            MultiGpuDecompositionStrategy::Temporal { steps_per_gpu } => {
                self.decompose_temporally(total_collocation_points, *steps_per_gpu)
            }
            MultiGpuDecompositionStrategy::Hybrid {
                spatial_dims,
                temporal_steps,
                overlap,
            } => self.decompose_hybrid(
                total_collocation_points,
                geometry_bounds,
                *spatial_dims,
                *temporal_steps,
                *overlap,
            ),
        }
    }
    /// Decompose spatially.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn decompose_spatially(
        &self,
        total_points: usize,
        bounds: (f64, f64, f64, f64),
        _dimensions: usize,
        overlap: f64,
    ) -> KwaversResult<Vec<WorkUnit>> {
        let n_gpus = (self.devices.shape()[0] * self.devices.shape()[1] * self.devices.shape()[2]);
        let points_per_gpu = total_points / n_gpus;
        let mut work_units = Vec::new();

        let (x_min, x_max, _y_min, _y_max) = bounds;

        for gpu_id in 0..n_gpus {
            let x_start = x_min + (x_max - x_min) * (gpu_id as f64) / (n_gpus as f64);
            let x_end = x_min + (x_max - x_min) * ((gpu_id + 1) as f64) / (n_gpus as f64);

            let _x_start_overlap = (x_start - overlap).max(x_min);
            let _x_end_overlap = (x_end + overlap).min(x_max);

            work_units.push(WorkUnit {
                id: gpu_id,
                device_id: gpu_id,
                complexity: points_per_gpu as f64,
                memory_required: points_per_gpu * std::mem::size_of::<f32>() * 4,
                priority: 1,
                dependencies: vec![],
                data_range: None,
                channel_range: None,
                sample_range: None,
            });
        }

        Ok(work_units)
    }
    /// Decompose temporally.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn decompose_temporally(
        &self,
        total_points: usize,
        _steps_per_gpu: usize,
    ) -> KwaversResult<Vec<WorkUnit>> {
        let n_gpus = (self.devices.shape()[0] * self.devices.shape()[1] * self.devices.shape()[2]);
        let mut work_units = Vec::new();

        for gpu_id in 0..n_gpus {
            work_units.push(WorkUnit {
                id: gpu_id,
                device_id: gpu_id,
                complexity: (total_points / n_gpus) as f64,
                memory_required: (total_points / n_gpus) * std::mem::size_of::<f32>() * 2,
                priority: 1,
                dependencies: if gpu_id > 0 { vec![gpu_id - 1] } else { vec![] },
                data_range: None,
                channel_range: None,
                sample_range: None,
            });
        }

        Ok(work_units)
    }
    /// Decompose hybrid.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub(super) fn decompose_hybrid(
        &self,
        total_points: usize,
        bounds: (f64, f64, f64, f64),
        spatial_dims: usize,
        temporal_steps: usize,
        overlap: f64,
    ) -> KwaversResult<Vec<WorkUnit>> {
        let spatial_work = self.decompose_spatially(total_points, bounds, spatial_dims, overlap)?;
        let temporal_work = self.decompose_temporally(total_points, temporal_steps)?;

        let mut work_units = Vec::new();
        let mut id_counter = 0;

        for spatial_unit in spatial_work {
            for temporal_unit in &temporal_work {
                work_units.push(WorkUnit {
                    id: id_counter,
                    device_id: spatial_unit.device_id,
                    complexity: spatial_unit.complexity * temporal_unit.complexity,
                    memory_required: spatial_unit.memory_required + temporal_unit.memory_required,
                    priority: 1,
                    dependencies: temporal_unit.dependencies.clone(),
                    data_range: spatial_unit.data_range.clone(),
                    channel_range: spatial_unit.channel_range.clone(),
                    sample_range: temporal_unit.sample_range.clone(),
                });
                id_counter += 1;
            }
        }

        Ok(work_units)
    }
}
