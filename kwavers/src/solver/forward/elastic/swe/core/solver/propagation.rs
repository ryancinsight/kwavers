//! Basic propagation methods for `ElasticWaveSolver`.

use super::super::super::integration::TimeIntegrator;
use super::super::super::types::{ElasticBodyForceConfig, ElasticWaveField};
use super::definition::ElasticWaveSolver;
use crate::core::error::{KwaversResult, NumericalError};
use crate::domain::sensor::recorder::fields::{SensorRecordField, SensorRecordSpec};
use crate::domain::sensor::recorder::simple::SensorRecorder;
use ndarray::{Array2, ArrayView2};

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

        // Multi-component recording (Phase A.2.5 of ADR 007):
        // allocate ux_data, uy_data, uz_data buffers in addition to the
        // pressure buffer (which carries the legacy uz-as-pressure trace
        // for back-compat with `extract_recorded_data` callers).
        let spec = SensorRecordSpec::from_fields(&[
            SensorRecordField::Pressure,
            SensorRecordField::VelocityX,
            SensorRecordField::VelocityY,
            SensorRecordField::VelocityZ,
        ]);
        self.sensor_recorder = SensorRecorder::with_spec(
            self.config.sensor_mask.as_ref(),
            (nx, ny, nz),
            recorded_steps,
            spec,
        )?;
        // Pre-collect velocity-source mask indices once, outside the time
        // loop, so per-step injection costs O(n_active) rather than O(N³).
        // Phase A.3 of ADR 007.
        let velocity_source_indices: Option<Vec<(usize, usize, usize)>> = self
            .config
            .velocity_source
            .as_ref()
            .filter(|vs| vs.has_any_component())
            .map(|vs| {
                vs.mask
                    .indexed_iter()
                    .filter_map(|(idx, &active)| active.then_some(idx))
                    .collect()
            });

        for step in 0..steps {
            integrator.step(&mut current_field, dt, body_force)?;
            current_field.time += dt;

            // ── Velocity-source injection (post-integrator Dirichlet hook) ──
            // After the velocity-Verlet step has updated vx/vy/vz from
            // acceleration, override at masked points with the supplied
            // signal sample for this step. The override is **assignment**
            // not addition, matching k-Wave's `Dirichlet` source mode for
            // velocity sources in pstdElastic2D / pstdElastic3D.
            if let (Some(ref vs), Some(ref active)) = (
                self.config.velocity_source.as_ref(),
                velocity_source_indices.as_ref(),
            ) {
                if let Some(ref ux_sig) = vs.ux_signal {
                    if let Some(&val) = ux_sig.as_slice().and_then(|s| s.get(step)) {
                        for &(i, j, k) in active.iter() {
                            current_field.vx[[i, j, k]] = val;
                        }
                    }
                }
                if let Some(ref uy_sig) = vs.uy_signal {
                    if let Some(&val) = uy_sig.as_slice().and_then(|s| s.get(step)) {
                        for &(i, j, k) in active.iter() {
                            current_field.vy[[i, j, k]] = val;
                        }
                    }
                }
                if let Some(ref uz_sig) = vs.uz_signal {
                    if let Some(&val) = uz_sig.as_slice().and_then(|s| s.get(step)) {
                        for &(i, j, k) in active.iter() {
                            current_field.vz[[i, j, k]] = val;
                        }
                    }
                }
            }

            if step % save_every == 0 {
                // Pressure-buffer entry: uz (legacy back-compat — many
                // existing callers consume `extract_recorded_data` which
                // returns the pressure buffer; this preserves their
                // contract since the elastic solver historically wrote uz
                // there).
                self.sensor_recorder.record_step(&current_field.uz)?;
                // Per-component displacement entries: ux_data / uy_data /
                // uz_data. `record_velocity_step` requires `record_step` to
                // have run first because it consumes `next_step - 1` as
                // the column index.
                self.sensor_recorder.record_velocity_step(
                    &current_field.ux,
                    &current_field.uy,
                    &current_field.uz,
                )?;
            }
        }
        Ok(current_field)
    }

    pub fn extract_recorded_data(&self) -> Option<Array2<f64>> {
        self.sensor_recorder.extract_pressure_data()
    }

    /// Per-component displacement traces recorded at sensor mask points.
    ///
    /// Returns `(ux_data, uy_data, uz_data)` where each is
    /// `Some(Array2<f64>)` of shape `(n_sensors, recorded_steps)` when the
    /// corresponding component buffer was allocated by the spec passed to
    /// the underlying `SensorRecorder`. The current `propagate` path
    /// allocates all three; this accessor exposes them through the public
    /// API for callers (e.g. PyO3 bridge) that cannot reach the
    /// `pub(crate)` `sensor_recorder` field directly.
    ///
    /// Phase A.2.5 of ADR 007.
    #[must_use]
    pub fn extract_recorded_displacement_components(
        &self,
    ) -> (Option<Array2<f64>>, Option<Array2<f64>>, Option<Array2<f64>>) {
        (
            self.sensor_recorder.extract_ux_data(),
            self.sensor_recorder.extract_uy_data(),
            self.sensor_recorder.extract_uz_data(),
        )
    }

    /// Borrow the full allocated sensor displacement buffer without cloning.
    #[must_use]
    pub fn recorded_data_view(&self) -> Option<ArrayView2<'_, f64>> {
        self.sensor_recorder.pressure_data_view()
    }

    /// Borrow only populated sensor displacement samples without cloning.
    #[must_use]
    pub fn recorded_data_prefix_view(&self) -> Option<ArrayView2<'_, f64>> {
        self.sensor_recorder.recorded_pressure_view()
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
