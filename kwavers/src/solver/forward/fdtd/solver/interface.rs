//! `solver::interface::Solver` trait bridge for the concrete FDTD solver.
//!
//! Maps the generic dispatcher contract (name/init/add_source/add_sensor/run/
//! field accessors/statistics/feature flags) onto the FDTD-specific
//! constructor + step loop. Initialisation is intentionally a no-op because
//! all heavy work happens in [`super::GenericFdtdSolver::new`].

use ndarray::Array3;
use std::sync::Arc;

use super::GenericFdtdSolver;
use crate::core::error::{ConfigError, KwaversError, KwaversResult};
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use crate::domain::sensor::GridSensorSet;
use crate::domain::source::Source;
use crate::solver::interface::{Solver, SolverFeature, SolverStatistics};

impl Solver for GenericFdtdSolver<Array3<f64>> {
    fn name(&self) -> &str {
        "FDTD"
    }

    fn initialize(&mut self, _grid: &Grid, _medium: &dyn Medium) -> KwaversResult<()> {
        // Core initialization happens in new(); this hook is reserved for re-init.
        Ok(())
    }

    fn add_source(&mut self, source: Box<dyn Source>) -> KwaversResult<()> {
        self.add_source_arc(Arc::from(source))
    }

    fn add_sensor(&mut self, _sensor: &GridSensorSet) -> KwaversResult<()> {
        // Map GridSensorSet to SensorRecorder logic
        // self.sensor_recorder.add_sensor(sensor);
        Ok(())
    }

    fn run(&mut self, num_steps: usize) -> KwaversResult<()> {
        for _ in 0..num_steps {
            self.step_forward()?;
        }
        Ok(())
    }

    fn step_forward(&mut self) -> KwaversResult<()> {
        // Inherent step_forward on GenericFdtdSolver; direct call, no run(1) dispatch.
        Self::step_forward(self)
    }

    fn pressure_field(&self) -> &Array3<f64> {
        &self.fields.p
    }

    fn recorded_sensor_pressure(&self) -> Option<ndarray::Array2<f64>> {
        self.sensor_recorder.extract_pressure_data()
    }

    fn velocity_fields(&self) -> (&Array3<f64>, &Array3<f64>, &Array3<f64>) {
        (&self.fields.ux, &self.fields.uy, &self.fields.uz)
    }

    fn statistics(&self) -> SolverStatistics {
        // Compute max pressure and velocity on the fly
        let max_pressure = self.fields.p.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
        let max_velocity = self
            .fields
            .ux
            .iter()
            .chain(self.fields.uy.iter())
            .chain(self.fields.uz.iter())
            .fold(0.0f64, |m, &v| m.max(v.abs()));

        SolverStatistics {
            total_steps: self.time_step_index,
            current_step: self.time_step_index,
            computation_time: std::time::Duration::default(), // Metrics need to track this
            memory_usage: 0,                                  // Estimator needed
            max_pressure,
            max_velocity,
        }
    }

    fn supports_feature(&self, feature: SolverFeature) -> bool {
        matches!(feature, SolverFeature::MultiThreaded)
            || (matches!(feature, SolverFeature::GpuAcceleration) && self.gpu_accelerator.is_some())
    }

    fn enable_feature(&mut self, feature: SolverFeature, enable: bool) -> KwaversResult<()> {
        match feature {
            SolverFeature::GpuAcceleration => {
                if enable && self.gpu_accelerator.is_none() {
                    return Err(KwaversError::Config(ConfigError::InvalidValue {
                        parameter: "enable_gpu_acceleration".to_owned(),
                        value: "true".to_owned(),
                        constraint: "GPU accelerator must be configured".to_owned(),
                    }));
                }
                self.config.enable_gpu_acceleration = enable;
                Ok(())
            }
            _ => Ok(()), // Ignore unsupported for now or error
        }
    }
}
