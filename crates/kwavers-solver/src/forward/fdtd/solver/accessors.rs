//! Public solver accessors: GPU accelerator hookup, CPML enable, CFL helpers,
//! metrics access/merge, sensor data extraction, orchestrated run loop.

use log::info;
use ndarray::{Array3, ArrayView2};
use std::sync::Arc;

use super::{FdtdGpuAccelerator, FdtdMetrics, GenericFdtdSolver};
use crate::forward::fdtd::config::KSpaceCorrectionMode;
use kwavers_boundary::cpml::{CPMLBoundary, CPMLConfig};
use kwavers_core::error::{KwaversError, KwaversResult};

impl GenericFdtdSolver<Array3<f64>> {
    pub fn set_gpu_accelerator(&mut self, accelerator: Arc<dyn FdtdGpuAccelerator>) {
        self.gpu_accelerator = Some(accelerator);
    }

    /// Enable C-PML boundary conditions.
    ///
    /// # Errors
    ///
    /// Returns [`KwaversError::InvalidInput`] when `kspace_correction =
    /// Spectral` is active. C-PML's convolutional memory update requires
    /// finite-difference gradients; spectral mode computes gradients entirely
    /// in k-space and never calls the CPML update, so enabling CPML with
    /// spectral mode is a silent no-op that hides a configuration mistake.
    ///
    /// **Resolution**: either set `kspace_correction = KSpaceCorrectionMode::None`
    /// (classical FD + CPML), or use a non-CPML absorbing layer (e.g. the
    /// kwavers multiplicative PML).
    ///
    /// **Reference**: Roden & Gedney (2000), *Microwave Opt. Technol. Lett.*
    /// 27(5), 334–339 — CPML assumes explicit FD curl operators.
    pub fn enable_cpml(
        &mut self,
        config: CPMLConfig,
        dt: f64,
        max_sound_speed: f64,
    ) -> KwaversResult<()> {
        // Invariant: Spectral k-space correction and CPML are mutually exclusive.
        // CPML's convolutional memory update modifies finite-difference gradient
        // arrays (dvx_scratch, dvy_scratch, etc.) that are absent in the spectral
        // path — the spectral update computes gradients via FFT and skips the CPML
        // correction branch entirely.  Permitting this combination would leave the
        // user with absorbing boundaries that do nothing while Spectral mode is on.
        if self.config.kspace_correction == KSpaceCorrectionMode::Spectral {
            return Err(KwaversError::InvalidInput(
                "KSpaceCorrectionMode::Spectral is incompatible with C-PML absorbing \
                 boundaries. CPML requires finite-difference gradient arrays that are \
                 not produced by the spectral path. Either set \
                 `kspace_correction = KSpaceCorrectionMode::None` or use a \
                 non-CPML absorbing layer."
                    .to_owned(),
            ));
        }
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

    /// Extract recorded sensor data as `Array2<f64>`
    /// Returns None if no sensors are configured or no data has been recorded
    pub fn extract_recorded_sensor_data(&self) -> Option<ndarray::Array2<f64>> {
        self.sensor_recorder
            .extract_pressure_data()
            .and_then(|data| data.try_into().ok())
    }

    /// Borrow the full allocated recorded sensor buffer without cloning.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn recorded_sensor_data_view(&self) -> Option<ArrayView2<'_, f64>> {
        self.sensor_recorder
            .pressure_data_view()
            .and_then(|view| view.try_into().ok())
    }

    /// Borrow only populated recorded sensor samples without cloning.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn recorded_sensor_prefix_view(&self) -> Option<ArrayView2<'_, f64>> {
        self.sensor_recorder
            .recorded_pressure_view()
            .and_then(|view| view.try_into().ok())
    }
    /// Run orchestrated.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
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
        Ok(self
            .sensor_recorder
            .extract_pressure_data()
            .and_then(|data| data.try_into().ok()))
    }
}

#[cfg(test)]
mod tests {
    use crate::forward::fdtd::config::{FdtdConfig, KSpaceCorrectionMode};
    use crate::forward::fdtd::FdtdSolver;
    use kwavers_boundary::cpml::CPMLConfig;
    use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
    use kwavers_core::error::KwaversError;
    use kwavers_grid::Grid;
    use kwavers_medium::HomogeneousMedium;
    use kwavers_source::GridSource;

    /// Construct a minimal FdtdSolver for accessor tests.
    ///
    /// Uses a 16³ grid with dx=dy=dz=1 mm, homogeneous water-like medium,
    /// no source, and a 4-cell CPML (fits inside the 16-cell grid).
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    fn build_solver(kspace_correction: KSpaceCorrectionMode) -> FdtdSolver {
        let grid = Grid::new(16, 16, 16, 1e-3, 1e-3, 1e-3).unwrap();
        let medium = HomogeneousMedium::new(
            DENSITY_WATER_NOMINAL,
            SOUND_SPEED_WATER_SIM,
            0.0,
            0.0,
            &grid,
        );
        let source = GridSource::new_empty();
        let config = FdtdConfig {
            kspace_correction,
            dt: 1e-7,
            nt: 10,
            ..FdtdConfig::default()
        };
        FdtdSolver::new(config, &grid, &medium, source).unwrap()
    }

    /// ## Theorem
    /// `enable_cpml` returns `KwaversError::InvalidInput` when
    /// `kspace_correction = KSpaceCorrectionMode::Spectral`.
    ///
    /// ## Proof
    /// CPML's convolutional update modifies finite-difference gradient arrays
    /// (dvx_scratch, etc.) that the spectral path does not produce.
    /// Permitting the combination silently bypasses the CPML corrections.
    /// The guard at the top of `enable_cpml` detects this state and returns
    /// `InvalidInput` before any boundary allocation occurs.
    /// # Panics
    /// - Panics with `"expected KwaversError::InvalidInput for Spectral+CPML, got {:?}"`.
    ///
    #[test]
    fn enable_cpml_rejects_spectral_kspace_correction() {
        let mut solver = build_solver(KSpaceCorrectionMode::Spectral);
        let cpml_config = CPMLConfig::with_thickness(4);
        let result = solver.enable_cpml(cpml_config, 1e-7, SOUND_SPEED_WATER_SIM);

        let err =
            result.expect_err("enable_cpml must return Err when kspace_correction = Spectral");
        let KwaversError::InvalidInput(ref msg) = err else {
            panic!(
                "expected KwaversError::InvalidInput for Spectral+CPML, got {:?}",
                err
            );
        };
        assert!(
            msg.contains("Spectral"),
            "error message must reference 'Spectral'; got: {msg}"
        );
        assert!(
            msg.contains("incompatible") || msg.contains("C-PML") || msg.contains("CPML"),
            "error message must reference the incompatibility; got: {msg}"
        );
    }

    /// ## Theorem
    /// `enable_cpml` returns `Ok(())` when
    /// `kspace_correction = KSpaceCorrectionMode::None`.
    ///
    /// ## Proof
    /// `None` mode uses finite-difference stencils; CPML gradient corrections
    /// are applied at each step.  The guard passes and `CPMLBoundary` is
    /// constructed from the supplied config, grid, and time step.
    /// # Panics
    /// - Panics if `enable_cpml must succeed when kspace_correction = None`.
    ///
    #[test]
    fn enable_cpml_accepts_none_kspace_correction() {
        let mut solver = build_solver(KSpaceCorrectionMode::None);
        let cpml_config = CPMLConfig::with_thickness(4);
        solver
            .enable_cpml(cpml_config, 1e-7, SOUND_SPEED_WATER_SIM)
            .expect("enable_cpml must succeed when kspace_correction = None");
        // Verify the boundary is now installed (cpml_boundary is Some).
        assert!(
            solver.cpml_boundary.is_some(),
            "cpml_boundary must be Some after successful enable_cpml"
        );
    }
}
