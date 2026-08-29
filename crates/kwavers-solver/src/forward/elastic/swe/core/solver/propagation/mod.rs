//! Basic propagation methods for `ElasticWaveSolver`.

use super::super::super::integration::integrator::PreparedBodyForces;
use super::super::super::integration::TimeIntegrator;
use super::super::super::scratch::ElasticStepScratch;
use super::super::super::types::{ElasticBodyForceConfig, ElasticWaveField};
use super::definition::ElasticWaveSolver;
use kwavers_core::error::{KwaversResult, ValidationError};
use kwavers_receiver::recorder::fields::{SensorRecordField, SensorRecordSpec};
use kwavers_receiver::recorder::simple::SensorRecorder;

mod plan;
mod sensors;
#[cfg(test)]
mod tests;

use plan::PropagationPlan;

impl ElasticWaveSolver {
    /// Propagate.
    /// # Errors
    /// - Returns a validation error when a field component shape or temporal
    ///   propagation parameter is invalid.
    /// - Propagates body-force, allocation, recording, and numerical errors.
    ///
    pub fn propagate(
        &mut self,
        initial_field: &ElasticWaveField,
        duration: f64,
        body_force: Option<&ElasticBodyForceConfig>,
    ) -> KwaversResult<ElasticWaveField> {
        let plan = PropagationPlan::for_field(
            &self.grid,
            &self.lambda,
            &self.mu,
            &self.density,
            &self.config,
            initial_field,
            duration,
        )?;
        let mut prepared_body_force = prepare_body_force(&self.grid, body_force)?;
        let mut current_field = initial_field.clone();
        let integrator =
            TimeIntegrator::new(&self.grid, &self.lambda, &self.mu, &self.density, &self.pml);
        let recorded_steps = plan.steps.div_ceil(plan.save_every);
        let (nx, ny, nz) = self.grid.dimensions();
        let mut scratch = ElasticStepScratch::new(nx, ny, nz);

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
                    .filter_map(|(idx, &active)| active.then_some((idx[0], idx[1], idx[2])))
                    .collect()
            });

        for step in 0..plan.steps {
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
                use crate::forward::elastic::swe::types::ElasticVelocitySourceMode;
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

            step_with_optional_prepared_force(
                &integrator,
                &mut current_field,
                plan.dt,
                prepared_body_force.as_mut(),
                &mut scratch,
            )?;
            current_field.time += plan.dt;

            if step % plan.save_every == 0 {
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
    /// - Returns a validation error when the displacement shape or configured
    ///   temporal propagation domain is invalid.
    /// - Propagates body-force, allocation, and numerical errors.
    ///
    pub fn propagate_waves(
        &self,
        initial_displacement: &leto::Array3<f64>,
    ) -> KwaversResult<Vec<ElasticWaveField>> {
        let (nx, ny, nz) = self.grid.dimensions();
        let expected_shape = [nx, ny, nz];
        let actual_shape = initial_displacement.shape();
        if actual_shape != expected_shape {
            return Err(ValidationError::DimensionMismatch {
                expected: format!("initial_displacement shape {expected_shape:?}"),
                actual: format!("{actual_shape:?}"),
            }
            .into());
        }
        let plan = PropagationPlan::for_initial_time(
            &self.grid,
            &self.lambda,
            &self.mu,
            &self.density,
            &self.config,
            0.0,
            self.config.simulation_time,
        )?;
        let mut initial_field = ElasticWaveField::new(nx, ny, nz);
        initial_field.uz.assign(initial_displacement);
        self.propagate_history_with_plan(&initial_field, plan, None)
    }

    /// Propagate waves with body force only override.
    /// # Errors
    /// - Returns a validation error when the configured temporal propagation
    ///   domain is invalid.
    /// - Propagates body-force, allocation, and numerical errors.
    ///
    pub fn propagate_waves_with_body_force_only_override(
        &self,
        body_force: Option<&ElasticBodyForceConfig>,
    ) -> KwaversResult<Vec<ElasticWaveField>> {
        let plan = PropagationPlan::for_initial_time(
            &self.grid,
            &self.lambda,
            &self.mu,
            &self.density,
            &self.config,
            0.0,
            self.config.simulation_time,
        )?;
        let (nx, ny, nz) = self.grid.dimensions();
        let initial_field = ElasticWaveField::new(nx, ny, nz);
        self.propagate_history_with_plan(&initial_field, plan, body_force)
    }

    fn propagate_history_with_plan(
        &self,
        initial_field: &ElasticWaveField,
        plan: PropagationPlan,
        body_force: Option<&ElasticBodyForceConfig>,
    ) -> KwaversResult<Vec<ElasticWaveField>> {
        let mut prepared_body_force = prepare_body_force(&self.grid, body_force)?;
        let mut current_field = initial_field.clone();
        let integrator =
            TimeIntegrator::new(&self.grid, &self.lambda, &self.mu, &self.density, &self.pml);
        let (nx, ny, nz) = self.grid.dimensions();
        let mut scratch = ElasticStepScratch::new(nx, ny, nz);
        let mut history = Vec::new();
        history.push(current_field.clone());
        for step_idx in 0..plan.steps {
            step_with_optional_prepared_force(
                &integrator,
                &mut current_field,
                plan.dt,
                prepared_body_force.as_mut(),
                &mut scratch,
            )?;
            current_field.time += plan.dt;
            if (step_idx + 1) % plan.save_every == 0 {
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

fn prepare_body_force(
    grid: &kwavers_grid::Grid,
    body_force: Option<&ElasticBodyForceConfig>,
) -> KwaversResult<Option<PreparedBodyForces>> {
    body_force
        .map(|force| PreparedBodyForces::new(grid, core::slice::from_ref(force)))
        .transpose()
}

fn step_with_optional_prepared_force(
    integrator: &TimeIntegrator<'_>,
    field: &mut ElasticWaveField,
    dt: f64,
    body_force: Option<&mut PreparedBodyForces>,
    scratch: &mut ElasticStepScratch,
) -> KwaversResult<()> {
    if let Some(body_force) = body_force {
        integrator.step_with_prepared_body_forces(field, dt, body_force, scratch)
    } else {
        integrator.step(field, dt, None, scratch)
    }
}

// Velocity-recording invariant: integration coverage lives in
// `external/elastic_julia_parity/compare_elastic.py` (matched-mode
// peak_ratio in [0.7, 1.4]) and `pykwavers/examples/
// ewp_elastic_2d_jl_compare.py`. A pure Rust unit test that constructs
// `ElasticWaveSolver` end-to-end is intentionally omitted here because the
// constructor's medium trait surface drifts frequently; the integration
// suite is the canonical guard.
