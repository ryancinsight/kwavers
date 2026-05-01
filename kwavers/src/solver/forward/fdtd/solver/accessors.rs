//! Public solver accessors: GPU accelerator hookup, CPML enable, CFL helpers,
//! metrics access/merge, sensor data extraction, orchestrated run loop.

use log::info;
use ndarray::Array3;
use std::sync::Arc;

use super::{FdtdGpuAccelerator, FdtdMetrics, GenericFdtdSolver};
use crate::core::error::KwaversResult;
use crate::domain::boundary::cpml::{CPMLBoundary, CPMLConfig};

impl GenericFdtdSolver<Array3<f64>> {
    pub fn set_gpu_accelerator(&mut self, accelerator: Arc<dyn FdtdGpuAccelerator>) {
        self.gpu_accelerator = Some(accelerator);
    }

    /// Enable C-PML boundary conditions
    pub fn enable_cpml(
        &mut self,
        config: CPMLConfig,
        dt: f64,
        max_sound_speed: f64,
    ) -> KwaversResult<()> {
        info!("Enabling C-PML boundary conditions");
        self.cpml_boundary = Some(CPMLBoundary::new_with_time_step(
            config,
            &self.grid,
            max_sound_speed,
            Some(dt),
        )?);
        Ok(())
    }

    /// Calculate maximum stable time step based on CFL condition
    pub fn max_stable_dt(&self, max_sound_speed: f64) -> f64 {
        let min_dx = self.grid.dx.min(self.grid.dy).min(self.grid.dz);
        let cfl_limit = self.spatial_order.cfl_limit();
        self.config.cfl_factor * cfl_limit * min_dx / max_sound_speed
    }

    /// Check if given timestep satisfies CFL condition
    pub fn check_cfl_stability(&self, dt: f64, max_sound_speed: f64) -> bool {
        let max_dt = self.max_stable_dt(max_sound_speed);
        dt <= max_dt
    }

    /// Get performance metrics
    pub fn get_metrics(&self) -> &FdtdMetrics {
        &self.metrics
    }

    /// Merge metrics from another solver instance
    pub fn merge_metrics(&mut self, other_metrics: &FdtdMetrics) {
        self.metrics.merge(other_metrics);
    }

    /// Extract recorded sensor data as Array2<f64>
    /// Returns None if no sensors are configured or no data has been recorded
    pub fn extract_recorded_sensor_data(&self) -> Option<ndarray::Array2<f64>> {
        self.sensor_recorder.extract_pressure_data()
    }

    pub fn run_orchestrated(
        &mut self,
        steps: usize,
    ) -> KwaversResult<Option<ndarray::Array2<f64>>> {
        // Record initial state t=0 to match k-Wave's convention (returning Nt+1 points)
        if self.time_step_index == 0 {
            self.sensor_recorder.record_step(&self.fields.p)?;
        }
        for _ in 0..steps {
            self.step_forward()?;
        }
        Ok(self.sensor_recorder.extract_pressure_data())
    }
}
