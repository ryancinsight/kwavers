//! Basic propagation methods for `ElasticWaveSolver`.

use super::super::super::integration::TimeIntegrator;
use super::super::super::types::{ElasticBodyForceConfig, ElasticWaveField};
use super::definition::ElasticWaveSolver;
use crate::core::error::{KwaversResult, NumericalError};
use crate::domain::sensor::recorder::fields::{SensorRecordField, SensorRecordSpec};
use crate::domain::sensor::recorder::simple::SensorRecorder;

mod sensors;

impl ElasticWaveSolver {
    /// Propagate.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
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
                "Calculated time step is non-positive".to_owned(),
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
            // ── Velocity-source injection (pre-integrator hook) ─────────────
            // Per the 2×2 mode-isolation study in
            // external/elastic_julia_parity/, Additive injection MUST happen
            // BEFORE the integrator step so the added v participates in the
            // velocity-Verlet u-update within the same step. Post-integrator
            // injection delays the Additive forcing by dt and produces
            // Pearson r ≈ 0.09 vs KWave.jl's pre-step Additive injection.
            //
            //   Dirichlet: vx[idx] = signal[step]   (assignment — pre-step is
            //                                         equivalent to post-step
            //                                         because the integrator
            //                                         re-derives v from a)
            //   Additive : vx[idx] += signal[step]  (forcing — must precede
            //                                         the integrator's u-update)
            //
            // Matches k-Wave's MATLAB `source.u_mode` semantics for elastic
            // PSTD solvers (where source injection is between the velocity
            // and stress half-steps in the stress-velocity formulation).
            if let (Some(vs), Some(active)) = (
                self.config.velocity_source.as_ref(),
                velocity_source_indices.as_ref(),
            ) {
                use crate::solver::forward::elastic::swe::types::ElasticVelocitySourceMode;
                let mode = vs.mode;
                if let Some(ref ux_sig) = vs.ux_signal {
                    if let Some(&val) = ux_sig.as_slice().and_then(|s| s.get(step)) {
                        match mode {
                            ElasticVelocitySourceMode::Dirichlet => {
                                for &(i, j, k) in active {
                                    current_field.vx[[i, j, k]] = val;
                                }
                            }
                            ElasticVelocitySourceMode::Additive => {
                                for &(i, j, k) in active {
                                    current_field.vx[[i, j, k]] += val;
                                }
                            }
                        }
                    }
                }
                if let Some(ref uy_sig) = vs.uy_signal {
                    if let Some(&val) = uy_sig.as_slice().and_then(|s| s.get(step)) {
                        match mode {
                            ElasticVelocitySourceMode::Dirichlet => {
                                for &(i, j, k) in active {
                                    current_field.vy[[i, j, k]] = val;
                                }
                            }
                            ElasticVelocitySourceMode::Additive => {
                                for &(i, j, k) in active {
                                    current_field.vy[[i, j, k]] += val;
                                }
                            }
                        }
                    }
                }
                if let Some(ref uz_sig) = vs.uz_signal {
                    if let Some(&val) = uz_sig.as_slice().and_then(|s| s.get(step)) {
                        match mode {
                            ElasticVelocitySourceMode::Dirichlet => {
                                for &(i, j, k) in active {
                                    current_field.vz[[i, j, k]] = val;
                                }
                            }
                            ElasticVelocitySourceMode::Additive => {
                                for &(i, j, k) in active {
                                    current_field.vz[[i, j, k]] += val;
                                }
                            }
                        }
                    }
                }
            }

            integrator.step(&mut current_field, dt, body_force)?;
            current_field.time += dt;

            if step % save_every == 0 {
                // Pressure-buffer entry: vz (legacy back-compat — many
                // existing callers consume `extract_recorded_data` which
                // returns the pressure buffer; this preserves their
                // contract since the elastic solver historically wrote uz
                // there).
                self.sensor_recorder.record_step(&current_field.vz)?;
                // Per-component VELOCITY entries: vx / vy / vz.
                // Particle velocity has a clear transient pulse when the
                // wave passes a sensor (displacement accumulates a DC
                // offset and cannot be used for timing detection).
                // `record_velocity_step` requires `record_step` to have
                // run first because it consumes `next_step - 1` as the
                // column index.
                self.sensor_recorder.record_velocity_step(
                    &current_field.vx,
                    &current_field.vy,
                    &current_field.vz,
                )?;
            }
        }
        Ok(current_field)
    }

    /// Propagate waves.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn propagate_waves(
        &self,
        initial_displacement: &ndarray::Array3<f64>,
    ) -> KwaversResult<Vec<ElasticWaveField>> {
        let (nx, ny, nz) = self.grid.dimensions();
        if initial_displacement.dim() != (nx, ny, nz) {
            return Err(NumericalError::InvalidOperation(
                "Initial displacement shape does not match grid".to_owned(),
            )
            .into());
        }
        let mut initial_field = ElasticWaveField::new(nx, ny, nz);
        initial_field.uz.assign(initial_displacement);
        self.propagate_history(&initial_field, self.config.simulation_time, None)
    }

    /// Propagate waves with body force only override.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn propagate_waves_with_body_force_only_override(
        &self,
        body_force: Option<&ElasticBodyForceConfig>,
    ) -> KwaversResult<Vec<ElasticWaveField>> {
        let (nx, ny, nz) = self.grid.dimensions();
        let initial_field = ElasticWaveField::new(nx, ny, nz);
        self.propagate_history(&initial_field, self.config.simulation_time, body_force)
    }

    /// Propagate history.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
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
                "Calculated time step is non-positive".to_owned(),
            )
            .into());
        }
        if !duration_s.is_finite() || duration_s <= 0.0 {
            return Err(NumericalError::InvalidOperation(
                "Simulation duration must be positive".to_owned(),
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

// Velocity-recording invariant: integration coverage lives in
// `external/elastic_julia_parity/compare_elastic.py` (matched-mode
// peak_ratio in [0.7, 1.4]) and `pykwavers/examples/
// ewp_elastic_2d_jl_compare.py`. A pure Rust unit test that constructs
// `ElasticWaveSolver` end-to-end is intentionally omitted here because the
// constructor's medium trait surface drifts frequently; the integration
// suite is the canonical guard.
