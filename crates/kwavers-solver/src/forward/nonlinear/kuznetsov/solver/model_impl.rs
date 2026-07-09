//! `AcousticWaveModel` trait implementation and conservation law checking.

use super::wave::KuznetsovWave;
use crate::forward::nonlinear::conservation::{ConservationDiagnostics, ViolationSeverity};
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_grid::Grid;
use kwavers_medium::Medium;
use kwavers_physics::traits::AcousticWaveModel;
use kwavers_source::Source;
use moirai_parallel::ParallelSliceMut;
use leto::{
    Array3,
    Array4,
};
use tracing::{error, info, warn};

impl AcousticWaveModel for KuznetsovWave {
    fn update_wave(
        &mut self,
        fields: &mut Array4<f64>,
        _prev_pressure: &Array3<f64>,
        source: &dyn Source,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
    ) -> KwaversResult<()> {
        if grid.nx != self.grid.nx || grid.ny != self.grid.ny || grid.nz != self.grid.nz {
            return Err(KwaversError::InvalidInput(
                "Grid dimensions mismatch with solver initialization".to_owned(),
            ));
        }

        if !medium.is_homogeneous() {
            warn!(
                "Kuznetsov heterogeneous media support is incomplete; nonlinear and diffusive terms use averaged properties"
            );
        }

        let mut pressure_field = fields.index_axis_mut(0, 0);

        if self.first_step {
            self.pressure_current.assign(&pressure_field);
            self.pressure_prev.assign(&pressure_field);

            self.compute_rhs(source, medium, t, dt);
            let rhs = &self.workspace.k1;

            {
                assert!(
                    pressure_field.is_standard_layout(),
                    "pressure_field must be C-contiguous (Array3 subview of C-contiguous Array4) for the migration"
                );
                let new_slice = pressure_field
                    .as_slice_mut()
                    .expect("pressure_field: standard-layout asserted just above; layout matched");
                assert!(
                    self.pressure_current.is_standard_layout(),
                    "pressure_current must be C-contiguous (default Array3 layout) for the migration"
                );
                let curr_slice = self
                    .pressure_current
                    .as_slice()
                    .expect("pressure_current: standard-layout asserted just above; layout matched");
                assert!(
                    rhs.is_standard_layout(),
                    "rhs must be C-contiguous (default Array3 layout) for the migration"
                );
                let rhs_slice = rhs
                    .as_slice()
                    .expect("rhs: standard-layout asserted just above; layout matched");
                new_slice.par_mut().enumerate(|idx, p_next: &mut f64| {
                    let p_curr = curr_slice[idx];
                    let accel = rhs_slice[idx];
                    *p_next = (0.5 * dt * dt).mul_add(accel, p_curr);
                });
            }

            self.workspace.update_time_history(&self.pressure_current);
            self.first_step = false;
        } else {
            self.compute_rhs(source, medium, t, dt);
            let rhs = &self.workspace.k1;

            {
                assert!(
                    pressure_field.is_standard_layout(),
                    "pressure_field must be C-contiguous (Array3 subview of C-contiguous Array4) for the migration"
                );
                let new_slice = pressure_field
                    .as_slice_mut()
                    .expect("pressure_field: standard-layout asserted just above; layout matched");
                assert!(
                    self.pressure_current.is_standard_layout(),
                    "pressure_current must be C-contiguous (default Array3 layout) for the migration"
                );
                let curr_slice = self
                    .pressure_current
                    .as_slice()
                    .expect("pressure_current: standard-layout asserted just above; layout matched");
                assert!(
                    self.pressure_prev.is_standard_layout(),
                    "pressure_prev must be C-contiguous (default Array3 layout) for the migration"
                );
                let prev_slice = self
                    .pressure_prev
                    .as_slice()
                    .expect("pressure_prev: standard-layout asserted just above; layout matched");
                assert!(
                    rhs.is_standard_layout(),
                    "rhs must be C-contiguous (default Array3 layout) for the migration"
                );
                let rhs_slice = rhs
                    .as_slice()
                    .expect("rhs: standard-layout asserted just above; layout matched");
                new_slice.par_mut().enumerate(|idx, p_next: &mut f64| {
                    let p_curr = curr_slice[idx];
                    let p_prev = prev_slice[idx];
                    let accel = rhs_slice[idx];
                    *p_next = (dt * dt).mul_add(accel, 2.0f64.mul_add(p_curr, -p_prev));
                });
            }

            self.pressure_prev.assign(&self.pressure_current);
            self.pressure_current.assign(&pressure_field);
            self.workspace.update_time_history(&self.pressure_current);
        }

        self.time_step_count += 1;
        self.current_time += dt;
        self.check_conservation_laws();

        Ok(())
    }

    fn report_performance(&self) {
        info!(
            equation_mode = ?self.config.equation_mode,
            grid_nx = self.grid.nx,
            grid_ny = self.grid.ny,
            grid_nz = self.grid.nz,
            time_steps_completed = self.time_step_count,
            nonlinearity_coefficient = self.config.nonlinearity_coefficient,
            acoustic_diffusivity = self.config.acoustic_diffusivity,
            "KuznetsovWave solver performance"
        );
    }

    fn set_nonlinearity_scaling(&mut self, scaling: f64) {
        self.nonlinearity_scaling = scaling;
    }
}

impl KuznetsovWave {
    /// Check conservation laws and log diagnostics.
    pub(super) fn check_conservation_laws(&mut self) {
        let should_check = self.conservation_tracker.as_ref().is_some_and(|tracker| {
            self.time_step_count
                .is_multiple_of(tracker.tolerances.check_interval)
        });

        if !should_check {
            return;
        }

        let (initial_energy, initial_momentum, initial_mass, tolerances) =
            if let Some(ref tracker) = self.conservation_tracker {
                (
                    tracker.initial_energy,
                    tracker.initial_momentum,
                    tracker.initial_mass,
                    tracker.tolerances,
                )
            } else {
                return;
            };

        let diagnostics = self.check_all_conservation(
            initial_energy,
            initial_momentum,
            initial_mass,
            self.time_step_count,
            self.current_time,
            &tolerances,
        );

        if let Some(ref mut tracker) = self.conservation_tracker {
            for diag in &diagnostics {
                if diag.severity > tracker.max_severity {
                    tracker.max_severity = diag.severity;
                }
            }
            tracker.history.extend(diagnostics.clone());
        }

        for diag in diagnostics {
            match diag.severity {
                ViolationSeverity::Acceptable => {}
                ViolationSeverity::Warning => {
                    warn!(%diag, "Kuznetsov conservation warning");
                }
                ViolationSeverity::Error => {
                    error!(%diag, "Kuznetsov conservation error");
                }
                ViolationSeverity::Critical => {
                    error!(%diag, "Kuznetsov conservation critical; solution may be physically invalid");
                }
            }
        }
    }
}
