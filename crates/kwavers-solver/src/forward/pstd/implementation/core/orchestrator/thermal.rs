//! PSTD acoustic→thermal coupling utilities.
//!
//! ## Physical model
//!
//! The volumetric heat source from acoustic absorption per Nyborg (1981):
//! ```text
//! Q = 2α c e    [W/m³]
//! e = P²/(2ρc²) + ½ρ|v|²   (acoustic energy density)
//! ```
//! Expanding: `Q = α·P²/(ρc) + αρc|v|²`.
//! For a plane wave: `|v| = P/(ρc)` → both terms contribute equally → `Q = 2α·P²/(ρc)`.
//!
//! ## Usage (standalone — single step caller)
//!
//! ```rust,ignore
//! let omega_c = TWO_PI * 1.0e6;   // 1 MHz center frequency
//! solver.populate_alpha_np_m_at_frequency(omega_c);
//! // After each acoustic step:
//! let q = solver.compute_acoustic_heat_source();
//! thermal_solver.update(&medium, &grid, dt_thermal, Some(q.view()))?;
//! ```
//!
//! ## Usage (coupled time loop)
//!
//! ```rust,ignore
//! let omega_c = TWO_PI * 1.0e6;
//! let rho_cp = 1000.0 * 3600.0; // kg/m³ × J/(kg·K)
//! solver.run_orchestrated_with_thermal(ThermalOrchestrationInput {
//!     acoustic_steps: time_steps,
//!     thermal: &mut thermal_solver,
//!     thermal_medium: &medium,
//!     omega_c,
//!     dt_thermal,
//!     n_acoustic_per_thermal,
//!     rho_cp,
//!     background_heat_ks: 0.0,
//! })?;
//! ```

use super::PSTDSolver;
use crate::forward::thermal_diffusion::ThermalDiffusionSolver;
use kwavers_core::error::KwaversResult;
use kwavers_medium::Medium;
use kwavers_physics::acoustics::conservation::acoustic_heat_source;
use kwavers_physics::acoustics::mechanics::absorption::AbsorptionMode;
use moirai_parallel::{enumerate_mut_with, Adaptive};
use leto::{
    Array2,
    Array3,
};
use std::{fmt, sync::Arc};

fn copy_scaled_absorption_field(dst: &mut Array3<f64>, alpha_si: &Array3<f64>, factor: f64) {
    assert_eq!(
        dst.shape(),
        alpha_si.shape(),
        "invariant: PSTD absorption destination shape matches alpha_si shape"
    );

    if let (Some(dst_values), Some(alpha_values)) = (
        dst.as_slice_mut(),
        alpha_si.as_slice(),
    ) {
        enumerate_mut_with::<Adaptive, _, _>(dst_values, |index, value| {
            *value = alpha_values[index] * factor;
        });
    } else {
        let [nx, ny, nz] = dst.shape();
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    dst[[i, j, k]] = alpha_si[[i, j, k]] * factor;
                }
            }
        }
    }
}

/// Input bundle for the coupled PSTD acoustic + thermal time loop.
///
/// The grouped input is a typed contract for the distinct physical units:
/// acoustic step count, angular frequency, thermal step length, coupling ratio,
/// volumetric heat capacity, and uniform background heating. Keeping these
/// values named prevents call-site unit transposition while preserving the
/// same acoustic and thermal update equations.
pub struct ThermalOrchestrationInput<'a> {
    /// Total number of acoustic time steps.
    pub acoustic_steps: usize,
    /// Mutable thermal diffusion solver updated every `n_acoustic_per_thermal`.
    pub thermal: &'a mut ThermalDiffusionSolver,
    /// Medium supplying `thermal_diffusivity` to the thermal solver.
    pub thermal_medium: &'a dyn Medium,
    /// Center angular frequency [rad/s] used once to populate `alpha_np_m`.
    pub omega_c: f64,
    /// Thermal time step [s].
    pub dt_thermal: f64,
    /// Acoustic steps between thermal updates. Must be at least one.
    pub n_acoustic_per_thermal: usize,
    /// Volumetric heat capacity `rho * cp` [J/(m^3 K)].
    pub rho_cp: f64,
    /// Uniform background heat rate [K/s] added at each thermal update.
    pub background_heat_ks: f64,
}

impl fmt::Debug for ThermalOrchestrationInput<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ThermalOrchestrationInput")
            .field("acoustic_steps", &self.acoustic_steps)
            .field("thermal", &"ThermalDiffusionSolver")
            .field("thermal_medium", &"dyn Medium")
            .field("omega_c", &self.omega_c)
            .field("dt_thermal", &self.dt_thermal)
            .field("n_acoustic_per_thermal", &self.n_acoustic_per_thermal)
            .field("rho_cp", &self.rho_cp)
            .field("background_heat_ks", &self.background_heat_ks)
            .finish()
    }
}

impl PSTDSolver {
    /// Scale the stored `α_SI` coefficients to physical Np/m at center frequency `omega_c`.
    ///
    /// ## Formula
    /// For `PowerLaw { y }`: `α(ω_c) = α_SI · ω_c^y`
    /// For `Stokes (y = 2)`:  `α(ω_c) = α_SI · ω_c²`
    /// For `Lossless`:        `α(ω_c) = 0`
    ///
    /// Call once after `PSTDSolver::new()` and before the first call to
    /// `compute_acoustic_heat_source()`. The result is stored in `alpha_np_m`
    /// and reused every acoustic step without further allocation.
    pub fn populate_alpha_np_m_at_frequency(&mut self, omega_c: f64) {
        // Extract the power-law exponent without holding a borrow on self.config.
        enum Mode {
            Lossless,
            Stokes,
            PowerLaw(f64),
        }
        let mode = match &self.config.absorption_mode {
            AbsorptionMode::Lossless => Mode::Lossless,
            AbsorptionMode::Stokes => Mode::Stokes,
            AbsorptionMode::PowerLaw { alpha_power, .. } => Mode::PowerLaw(*alpha_power),
            _ => Mode::Lossless,
        };

        match mode {
            Mode::Lossless => {
                // Lossless ⟹ no thermal heating; release memory if previously allocated.
                self.alpha_np_m = None;
            }
            Mode::Stokes => {
                // y = 2: α(ω_c) = α_SI · ω_c²
                if let Some(ref kernel) = self.absorption {
                    let omega_sq = omega_c * omega_c;
                    let alpha_si = &kernel.alpha_si;
                    let shape = alpha_si.shape();
                    let buf = self.alpha_np_m.get_or_insert_with(|| Array3::zeros(shape));
                    copy_scaled_absorption_field(buf, alpha_si, omega_sq);
                }
            }
            Mode::PowerLaw(y) => {
                if let Some(ref kernel) = self.absorption {
                    let omega_y = omega_c.powf(y);
                    let alpha_si = &kernel.alpha_si;
                    let shape = alpha_si.shape();
                    let buf = self.alpha_np_m.get_or_insert_with(|| Array3::zeros(shape));
                    copy_scaled_absorption_field(buf, alpha_si, omega_y);
                }
            }
        }
    }

    /// Override `alpha_np_m` with an externally computed per-cell absorption field [Np/m].
    ///
    /// Use when the center-frequency absorption is known a priori (e.g., derived from
    /// CT/MRI tissue segmentation) rather than from the simulation absorption mode.
    pub fn set_alpha_np_m(&mut self, alpha: Array3<f64>) {
        self.alpha_np_m = Some(alpha);
    }

    /// Read-only view of the stored per-cell absorption coefficient [Np/m].
    ///
    /// Returns `None` when thermal coupling has not been requested (lossless path or
    /// before `populate_alpha_np_m_at_frequency` / `set_alpha_np_m` is called).
    #[must_use]
    pub fn alpha_np_m(&self) -> Option<leto::ArrayView3<'_, f64>> {
        self.alpha_np_m.as_ref().map(|a| a.view())
    }

    /// Compute the volumetric acoustic heat source Q [W/m³] from current field state.
    ///
    /// Uses `alpha_np_m` populated by `populate_alpha_np_m_at_frequency()` or
    /// `set_alpha_np_m()`. Returns an all-zero array when `alpha_np_m` is `None`
    /// (lossless simulation or before thermal coupling is initialised).
    ///
    /// Pass the returned array to `ThermalDiffusionSolver::update()` as `external_source`.
    #[must_use]
    pub fn compute_acoustic_heat_source(&self) -> Array3<f64> {
        let [nx, ny, nz] = self.fields.p.shape();
        match self.alpha_np_m.as_ref() {
            None => Array3::zeros((nx, ny, nz)),
            Some(alpha) => {
                let alpha_leto = alpha.clone();
                acoustic_heat_source(
                    &self.fields.p,
                    &self.fields.ux,
                    &self.fields.uy,
                    &self.fields.uz,
                    &self.materials.rho0,
                    &self.materials.c0,
                    &alpha_leto,
                )
                .try_into()
                .expect("thermal heat source must convert to ndarray")
            }
        }
    }

    /// Run a coupled acoustic + thermal time loop.
    ///
    /// ## Parameters
    ///
    /// - `acoustic_steps`: total number of acoustic time steps.
    /// - `thermal`: mutable thermal diffusion solver; updated every
    ///   `n_acoustic_per_thermal` acoustic steps.
    /// - `thermal_medium`: medium supplying `thermal_diffusivity` to the thermal solver.
    /// - `omega_c`: center angular frequency [rad/s]; used once to populate `alpha_np_m`.
    /// - `dt_thermal`: thermal time step [s].  Typically
    ///   `n_acoustic_per_thermal * dt_acoustic`.
    /// - `n_acoustic_per_thermal`: number of acoustic steps between thermal updates.
    ///   Must be ≥ 1.
    /// - `rho_cp`: volumetric heat capacity ρ·cp [J/(m³·K)] of the thermal medium;
    ///   converts Q [W/m³] → K/s for the thermal solver.
    /// - `background_heat_ks`: uniform background heat rate [K/s] added at every thermal
    ///   step (e.g. metabolic heat Q_m/(ρ·cp)). Pass `0.0` for acoustic-only heating.
    ///
    /// ## Returns
    ///
    /// Acoustic sensor data (same as `run_orchestrated`).
    ///
    /// ## Errors
    ///
    /// Propagates `KwaversError` from acoustic or thermal solver steps.
    pub fn run_orchestrated_with_thermal(
        &mut self,
        input: ThermalOrchestrationInput<'_>,
    ) -> KwaversResult<Option<Array2<f64>>> {
        debug_assert!(
            input.n_acoustic_per_thermal >= 1,
            "n_acoustic_per_thermal must be >= 1"
        );
        self.populate_alpha_np_m_at_frequency(input.omega_c);

        if self.time_step_index == 0 && self.source_handler.has_initial_pressure() {
            self.sensor_recorder.record_step(&self.fields.p)?;
        }

        // Clone Arc so we can hold &Grid independently of &mut self during the loop.
        let grid = Arc::clone(&self.grid);

        for i in 0..input.acoustic_steps {
            self.step_forward()?;

            if (i + 1) % input.n_acoustic_per_thermal == 0 {
                let q_wm3 = self.compute_acoustic_heat_source();
                // Q [W/m³] → K/s; add uniform background (metabolic heat etc.).
                let q_ks = q_wm3.mapv(|v| v / input.rho_cp + input.background_heat_ks);
                input.thermal.update(
                    input.thermal_medium,
                    &grid,
                    input.dt_thermal,
                    Some(q_ks.view()),
                )?;
            }
        }

        Ok(self
            .sensor_recorder
            .extract_pressure_data()
            .and_then(|data| data.try_into().ok()))
    }
}
