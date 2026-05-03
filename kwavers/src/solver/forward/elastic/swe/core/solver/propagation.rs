//! Basic propagation methods for `ElasticWaveSolver`.

use super::super::super::integration::TimeIntegrator;
use super::super::super::types::{ElasticBodyForceConfig, ElasticWaveField};
use super::definition::ElasticWaveSolver;
use crate::core::error::{KwaversResult, NumericalError};
use crate::domain::sensor::recorder::simple::SensorRecorder;
use ndarray::Array2;

impl ElasticWaveSolver {
    pub fn propagate(
        &mut self,
        initial_field: &ElasticWaveField,
        duration: f64,
        body_force: Option<&ElasticBodyForceConfig>,
    ) -> KwaversResult<ElasticWaveField> {
        let mut current_field = initial_field.clone();
        let integrator = TimeIntegrator::new(
            &self.grid,
            &self.lambda,
            &self.mu,
            &self.density,
            self.pml.attenuation_field(),
        );
        let dt = if self.config.time_step > 0.0 {
            self.config.time_step
        } else {
            integrator.calculate_stable_timestep(self.config.cfl_factor)
        };
        if dt <= 0.0 {
            return Err(NumericalError::InvalidOperation(
                "Calculated time step is non-positive".to_string(),
            )
            .into());
        }
        let steps = (duration / dt).ceil() as usize;
        let save_every = self.config.save_every.max(1);
        let recorded_steps = steps.div_ceil(save_every);
        let (nx, ny, nz) = self.grid.dimensions();
        self.sensor_recorder = SensorRecorder::new(
            self.config.sensor_mask.as_ref(),
            (nx, ny, nz),
            recorded_steps,
        )?;
        for step in 0..steps {
            integrator.step(&mut current_field, dt, body_force)?;
            current_field.time += dt;
            if step % save_every == 0 {
                self.sensor_recorder.record_step(&current_field.uz)?;
            }
        }
        Ok(current_field)
    }

    pub fn extract_recorded_data(&self) -> Option<Array2<f64>> {
        self.sensor_recorder.extract_pressure_data()
    }

    pub fn propagate_waves(
        &self,
        initial_displacement: &ndarray::Array3<f64>,
    ) -> KwaversResult<Vec<ElasticWaveField>> {
        let (nx, ny, nz) = self.grid.dimensions();
        if initial_displacement.dim() != (nx, ny, nz) {
            return Err(NumericalError::InvalidOperation(
                "Initial displacement shape does not match grid".to_string(),
            )
            .into());
        }
        let mut initial_field = ElasticWaveField::new(nx, ny, nz);
        initial_field.uz.assign(initial_displacement);
        self.propagate_history(&initial_field, self.config.simulation_time, None)
    }

    pub fn propagate_waves_with_body_force_only_override(
        &self,
        body_force: Option<&ElasticBodyForceConfig>,
    ) -> KwaversResult<Vec<ElasticWaveField>> {
        let (nx, ny, nz) = self.grid.dimensions();
        let initial_field = ElasticWaveField::new(nx, ny, nz);
        self.propagate_history(&initial_field, self.config.simulation_time, body_force)
    }

    pub(super) fn propagate_history(
        &self,
        initial_field: &ElasticWaveField,
        duration_s: f64,
        body_force: Option<&ElasticBodyForceConfig>,
    ) -> KwaversResult<Vec<ElasticWaveField>> {
        let mut current_field = initial_field.clone();
        let integrator = TimeIntegrator::new(
            &self.grid,
            &self.lambda,
            &self.mu,
            &self.density,
            self.pml.attenuation_field(),
        );
        let dt = if self.config.time_step > 0.0 {
            self.config.time_step
        } else {
            integrator.calculate_stable_timestep(self.config.cfl_factor)
        };
        if dt <= 0.0 {
            return Err(NumericalError::InvalidOperation(
                "Calculated time step is non-positive".to_string(),
            )
            .into());
        }
        if !duration_s.is_finite() || duration_s <= 0.0 {
            return Err(NumericalError::InvalidOperation(
                "Simulation duration must be positive".to_string(),
            )
            .into());
        }
        let save_every = self.config.save_every.max(1);
        let steps = (duration_s / dt).ceil() as usize;
        let mut history = Vec::new();
        history.push(current_field.clone());
        for step_idx in 0..steps {
            integrator.step(&mut current_field, dt, body_force)?;
            current_field.time += dt;
            if (step_idx + 1) % save_every == 0 {
                history.push(current_field.clone());
            }
        }
        let needs_final = match history.last() {
            None => true,
            Some(f) => f.time != current_field.time,
        };
        if needs_final {
            history.push(current_field.clone());
        }
        Ok(history)
    }
}
