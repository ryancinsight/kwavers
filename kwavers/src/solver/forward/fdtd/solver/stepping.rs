//! Yee leapfrog `step_forward` and debug-only NaN scans.
//!
//! Velocity update precedes pressure update (matches `t^{n+½}` velocity ↔
//! `t^{n+1}` pressure staggering). Source injection happens after each
//! field update so distributed sources contribute additively and Dirichlet
//! masks override the staggered solve.

use ndarray::Array3;

use super::GenericFdtdSolver;
use crate::core::error::KwaversResult;

impl GenericFdtdSolver<Array3<f64>> {
    /// Perform a single time step.
    ///
    /// In debug builds, full-field NaN scans are performed after each sub-step
    /// to catch numerical instabilities early. In release builds, these scans
    /// are elided for performance (O(N) per scan × 4 scans per step).
    #[inline]
    pub fn step_forward(&mut self) -> KwaversResult<()> {
        let time_index = self.time_step_index;
        let dt = self.config.dt;

        // 1. Update Velocity (from current pressure field)
        self.update_velocity(dt)?;
        #[cfg(debug_assertions)]
        self.check_nan_velocity(time_index, "update_velocity")?;

        // 2. Inject Force Sources
        if self.source_handler.has_velocity_source() {
            self.source_handler.inject_force_source(
                time_index,
                &mut self.fields.ux,
                &mut self.fields.uy,
                &mut self.fields.uz,
            );
        }

        self.apply_dynamic_velocity_sources(dt);
        #[cfg(debug_assertions)]
        self.check_nan_velocity(time_index, "dynamic_velocity_sources")?;

        // 3. Update Pressure
        self.update_pressure(dt)?;
        #[cfg(debug_assertions)]
        self.check_nan_pressure(time_index, "update_pressure")?;

        // 4. Apply pressure sources after update (additive + Dirichlet enforcement)
        if self.source_handler.has_pressure_source() {
            self.source_handler
                .inject_pressure_source(time_index, &mut self.fields.p);
        }
        self.apply_dynamic_pressure_sources(dt);
        self.source_handler
            .enforce_pressure_dirichlet(time_index, &mut self.fields.p);
        self.apply_dynamic_pressure_dirichlet(dt);

        #[cfg(debug_assertions)]
        self.check_nan_pressure(time_index, "pressure_sources")?;

        // 5. Apply Boundary (CPML is applied within updates via self.cpml_boundary)

        // 6. Record Sensors
        self.sensor_recorder.record_step(&self.fields.p)?;

        self.time_step_index += 1;

        Ok(())
    }

    /// Check velocity fields for NaN values (debug-only).
    ///
    /// Returns `KwaversError::Numerical(NaN)` instead of panicking, enabling
    /// upstream callers to handle instabilities gracefully (e.g., reduce dt,
    /// log diagnostics, or return partial results).
    #[cfg(debug_assertions)]
    pub(super) fn check_nan_velocity(&self, step: usize, phase: &str) -> KwaversResult<()> {
        use crate::core::error::{KwaversError, NumericalError};
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
    #[cfg(debug_assertions)]
    pub(super) fn check_nan_pressure(&self, step: usize, phase: &str) -> KwaversResult<()> {
        use crate::core::error::{KwaversError, NumericalError};
        if self.fields.p.iter().any(|&x| x.is_nan()) {
            return Err(KwaversError::Numerical(NumericalError::NaN {
                operation: format!("FDTD {phase} at step {step}"),
                inputs: "pressure field contains NaN".to_string(),
            }));
        }
        Ok(())
    }
}
