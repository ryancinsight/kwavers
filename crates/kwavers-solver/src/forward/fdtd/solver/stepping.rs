//! Yee leapfrog `step_forward` and debug-only NaN scans.
//!
//! Velocity update precedes pressure update (matches `t^{n+½}` velocity ↔
//! `t^{n+1}` pressure staggering). Source injection happens after each
//! field update so distributed sources contribute additively and Dirichlet
//! masks override the staggered solve.

use leto::Array3;

use super::GenericFdtdSolver;
use crate::forward::fdtd::config::TemporalScheme;
use kwavers_core::error::KwaversResult;
#[cfg(debug_assertions)]
use kwavers_source::SourceField;

impl GenericFdtdSolver<Array3<f64>> {
    /// Perform a single time step.
    ///
    /// In debug builds, full-field NaN scans are performed after each completed
    /// velocity and pressure phase. When sources can mutate a phase, an
    /// additional pre-source scan preserves propagation-versus-source failure
    /// attribution. Release builds elide every scan.
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    #[inline]
    pub fn step_forward(&mut self) -> KwaversResult<()> {
        let time_index = self.time_step_index;
        let dt = self.config.dt;

        // Fourth-order composition replaces only the *propagation*: sources are
        // injected once per step below, not once per sub-step, because they
        // model an external drive at this instant rather than part of the
        // Hamiltonian flow being composed.
        if self.config.temporal_scheme == TemporalScheme::Yoshida4 {
            self.advance_yoshida4(dt)?;
            self.finish_velocity_phase(time_index, dt, "yoshida4")?;
            return self.finish_step(time_index, dt, "yoshida4");
        }

        // 1. Update Velocity (from current pressure field)
        self.update_velocity(dt)?;
        self.finish_velocity_phase(time_index, dt, "update_velocity")?;

        // 3. Update Pressure
        self.update_pressure(dt)?;

        self.finish_step(time_index, dt, "update_pressure")
    }

    /// Inject velocity sources once after the selected propagation phase.
    fn finish_velocity_phase(
        &mut self,
        time_index: usize,
        dt: f64,
        propagation_phase: &str,
    ) -> KwaversResult<()> {
        #[cfg(not(debug_assertions))]
        let _ = propagation_phase;
        let has_grid_source = self.source_handler.has_velocity_source();
        #[cfg(debug_assertions)]
        let sources_may_mutate = has_grid_source
            || self.dynamic_sources.iter().any(|(source, _)| {
                matches!(
                    source.source_type(),
                    SourceField::VelocityX | SourceField::VelocityY | SourceField::VelocityZ
                )
            });

        #[cfg(debug_assertions)]
        if sources_may_mutate {
            self.check_nan_velocity(time_index, propagation_phase)?;
        }

        if has_grid_source {
            self.source_handler.inject_force_source(
                time_index,
                &mut self.fields.ux,
                &mut self.fields.uy,
                &mut self.fields.uz,
            );
        }
        self.apply_dynamic_velocity_sources(dt);

        #[cfg(debug_assertions)]
        self.check_nan_velocity(
            time_index,
            if sources_may_mutate {
                "velocity_sources"
            } else {
                propagation_phase
            },
        )?;

        Ok(())
    }

    /// Pressure sources, sensor recording and the step counter — the part of a
    /// step that is not propagation, shared by both time-integration schemes.
    fn finish_step(
        &mut self,
        time_index: usize,
        dt: f64,
        propagation_phase: &str,
    ) -> KwaversResult<()> {
        #[cfg(not(debug_assertions))]
        let _ = propagation_phase;
        let has_grid_source = self.source_handler.has_pressure_source();
        #[cfg(debug_assertions)]
        let sources_may_mutate = has_grid_source
            || self
                .dynamic_sources
                .iter()
                .any(|(source, _)| source.source_type() == SourceField::Pressure);

        #[cfg(debug_assertions)]
        if sources_may_mutate {
            self.check_nan_pressure(time_index, propagation_phase)?;
        }

        if has_grid_source {
            self.source_handler
                .inject_pressure_source(time_index, &mut self.fields.p);
        }
        self.apply_dynamic_pressure_sources(dt);
        self.source_handler
            .enforce_pressure_dirichlet(time_index, &mut self.fields.p);
        self.apply_dynamic_pressure_dirichlet(dt);

        #[cfg(debug_assertions)]
        self.check_nan_pressure(
            time_index,
            if sources_may_mutate {
                "pressure_sources"
            } else {
                propagation_phase
            },
        )?;

        // CPML is applied within the updates via `self.cpml_boundary`.
        self.sensor_recorder.record_step(&self.fields.p)?;
        self.time_step_index += 1;

        Ok(())
    }

    /// Advance the propagation by `dt` with Yoshida's fourth-order composition.
    ///
    /// Three sub-steps at `w1·dt`, `w0·dt`, `w1·dt` with `2w1 + w0 = 1`, each
    /// the self-adjoint `K(h/2) D(h) K(h/2)`. Composing the plain
    /// kick-then-drift step instead would gain no order, because Yoshida's
    /// cancellation needs a self-adjoint base method.
    fn advance_yoshida4(&mut self, dt: f64) -> KwaversResult<()> {
        let cbrt2 = 2.0_f64.cbrt();
        let denominator = 2.0 - cbrt2;
        let w1 = 1.0 / denominator;
        let w0 = -cbrt2 / denominator;
        debug_assert!(
            (2.0 * w1 + w0 - 1.0).abs() < 1e-12,
            "invariant: Yoshida weights advance the state by exactly dt"
        );

        for weight in [w1, w0, w1] {
            let h = weight * dt;
            self.update_velocity(0.5 * h)?;
            self.update_pressure(h)?;
            self.update_velocity(0.5 * h)?;
        }
        Ok(())
    }

    /// Check velocity fields for NaN values (debug-only).
    ///
    /// Returns `KwaversError::Numerical(NaN)` instead of panicking, enabling
    /// upstream callers to handle instabilities gracefully (e.g., reduce dt,
    /// log diagnostics, or return partial results).
    /// # Errors
    /// - Returns [`crate::KwaversError::Numerical`] if the precondition for a Numerical-class constraint is violated.
    ///
    #[cfg(debug_assertions)]
    pub(super) fn check_nan_velocity(&self, step: usize, phase: &str) -> KwaversResult<()> {
        use kwavers_core::error::{KwaversError, NumericalError};
        for (name, field) in [
            ("ux", &self.fields.ux),
            ("uy", &self.fields.uy),
            ("uz", &self.fields.uz),
        ] {
            if field.iter().any(|&x| x.is_nan()) {
                return Err(KwaversError::Numerical(NumericalError::NaN {
                    operation: format!("FDTD {phase} at step {step}"),
                    inputs: format!("field {name} contains NaN"),
                }));
            }
        }
        Ok(())
    }

    /// Check pressure field for NaN values (debug-only).
    /// # Errors
    /// - Returns [`crate::KwaversError::Numerical`] if the precondition for a Numerical-class constraint is violated.
    ///
    #[cfg(debug_assertions)]
    pub(super) fn check_nan_pressure(&self, step: usize, phase: &str) -> KwaversResult<()> {
        use kwavers_core::error::{KwaversError, NumericalError};
        if self.fields.p.iter().any(|&x| x.is_nan()) {
            return Err(KwaversError::Numerical(NumericalError::NaN {
                operation: format!("FDTD {phase} at step {step}"),
                inputs: "pressure field contains NaN".to_owned(),
            }));
        }
        Ok(())
    }
}
