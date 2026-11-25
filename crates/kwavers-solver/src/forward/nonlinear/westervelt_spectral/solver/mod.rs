//! Main Westervelt equation solver implementation
//!
//! # Physics: Westervelt Nonlinear Wave Equation
//!
//! ## Theorem (Westervelt 1963)
//!
//! For a progressive plane wave in a thermoviscous fluid, cumulative nonlinear
//! distortion and absorption are governed by the Westervelt equation:
//! ```text
//!   ∇²p − (1/c₀²) ∂²p/∂t² + (δ/c₀⁴) ∂³p/∂t³ + (β/ρ₀c₀⁴) ∂²p²/∂t² = 0
//! ```
//! where:
//! - `c₀` = small-signal sound speed [m/s]
//! - `ρ₀` = ambient density [kg/m³]
//! - `δ` = diffusivity of sound [m²/s] (thermoviscous absorption coefficient)
//! - `β = 1 + B/(2A)` = coefficient of nonlinearity (dimensionless)
//!
//! ## Spectral (k-Space) Discretization
//!
//! The Laplacian `∇²p` is computed spectrally via the DFT:
//! ```text
//!   ∇²p = IFFT(−|k|² · FFT(p))
//! ```
//! The operator stores the three wavenumber axes and evaluates
//! `|k|² = kx² + ky² + kz²` on the Hermitian half spectrum. Each resolved
//! Fourier mode has Laplacian eigenvalue `−|k|²`.
//!
//! ## Time-Stepping (Leapfrog)
//!
//! A 3-level leapfrog scheme is used for the second-order time derivative:
//! ```text
//!   p^{n+1} = 2p^n − p^{n−1} + (c₀Δt)² · ∇²p^n
//!           + Δt² δ ∇²_FD[(p^n − p^{n−1})/Δt]        [thermoviscous]
//!           + (βΔt²/ρ₀c₀²) · ∂²(p²)/∂t²              [nonlinear]
//! ```
//! Stability requires the acoustic CFL condition: `c₀ · Δt · |k|_max ≤ 2`.
//!
//! ## References
//! - Westervelt, P.J. (1963). Parametric acoustic array. J. Acoust. Soc. Am. 35(4), 535–537.
//! - Hamilton, M.F. & Blackstock, D.T. (1998). Nonlinear Acoustics. Academic Press, Ch. 3.

use crate::forward::nonlinear::conservation::{
    ConservationDiagnostics, ConservationTolerances, ConservationTracker, ViolationSeverity,
};
use crate::forward::nonlinear::spectral::PeriodicLaplacian;
use kwavers_grid::Grid;
use kwavers_medium::Medium;
use leto::Array3;
use log::warn;
use std::sync::{Arc, Mutex};

use super::metrics::WesterveltStepMetrics;

mod conservation;
mod wave_model;

/// Cached medium properties for conservation calculations
#[derive(Debug, Clone)]
pub(super) struct MediumProperties {
    pub(super) rho0: f64,
    pub(super) c0: f64,
}

/// Westervelt equation solver with proper second-order time derivatives
#[derive(Debug)]
pub struct WesterveltWave {
    laplacian: PeriodicLaplacian,
    pub(super) nonlinearity_scaling: f64,
    /// Scales the viscoelastic damping term `δ∇²(∂p/∂t)`. Default 1.0; set to
    /// 0.0 for a lossless linear run. NOTE: the explicit FD discretization of this
    /// term is only conditionally stable (its high-wavenumber amplification grows
    /// as `dt·δ·k²`); lossy runs require a damping-CFL-safe `dt` or a k-space
    /// treatment. See ADR/spawn task on Westervelt absorption stability.
    pub(super) damping_scaling: f64,
    /// Pressure history using buffer rotation for zero-allocation updates.
    pub(super) pressure_buffers: [Array3<f64>; 3],
    pub(super) buffer_indices: [usize; 3], // [next, current, previous]
    /// Real Laplacian read by the leapfrog update.
    laplacian_scratch: Array3<f64>,
    /// Pre-allocated nonlinear-term workspace.
    pub(super) nonlinear_scratch: Array3<f64>,
    /// Pre-allocated viscoelastic damping workspace.
    pub(super) damping_scratch: Array3<f64>,
    /// Pre-allocated source-mask workspace.
    pub(super) source_mask_scratch: Array3<f64>,
    pub(super) metrics: Arc<Mutex<WesterveltStepMetrics>>,
    pub(super) conservation_tracker: Option<ConservationTracker>,
    pub(super) current_step: usize,
    pub(super) current_time: f64,
    pub(super) grid_cache: Option<Grid>,
    pub(super) medium_properties: Option<MediumProperties>,
}

impl WesterveltWave {
    /// Create a new Westervelt solver.
    pub fn new(grid: &Grid) -> Self {
        let shape = (grid.nx, grid.ny, grid.nz);

        Self {
            laplacian: PeriodicLaplacian::new(grid),
            nonlinearity_scaling: 1.0,
            damping_scaling: 1.0,
            pressure_buffers: [
                Array3::zeros(shape),
                Array3::zeros(shape),
                Array3::zeros(shape),
            ],
            buffer_indices: [0, 1, 2],
            laplacian_scratch: Array3::zeros(shape),
            nonlinear_scratch: Array3::zeros(shape),
            damping_scratch: Array3::zeros(shape),
            source_mask_scratch: grid.array_zeros(),
            metrics: Arc::new(Mutex::new(WesterveltStepMetrics::new())),
            conservation_tracker: None,
            current_step: 0,
            current_time: 0.0,
            grid_cache: Some(grid.clone()),
            medium_properties: None,
        }
    }

    /// Enable conservation diagnostics with specified tolerances.
    pub fn enable_conservation_diagnostics(
        &mut self,
        tolerances: ConservationTolerances,
        medium: &dyn Medium,
    ) {
        if let Some(ref grid) = self.grid_cache {
            let (center_x, center_y, center_z) = grid.center();
            let rho0 =
                kwavers_medium::density_at(medium, center_x, center_y, center_z, grid).into_base();
            let c0 = kwavers_medium::sound_speed_at(medium, center_x, center_y, center_z, grid)
                .into_base();
            self.medium_properties = Some(MediumProperties { rho0, c0 });
        }

        let initial_energy = self.calculate_total_energy();
        let initial_momentum = self.calculate_total_momentum();
        let initial_mass = self.calculate_total_mass();

        self.conservation_tracker = Some(ConservationTracker::new(
            initial_energy,
            initial_momentum,
            initial_mass,
            tolerances,
        ));
    }

    /// Disable conservation diagnostics.
    pub fn disable_conservation_diagnostics(&mut self) {
        self.conservation_tracker = None;
    }

    /// Get conservation diagnostic summary.
    pub fn get_conservation_summary(&self) -> Option<String> {
        self.conservation_tracker
            .as_ref()
            .map(|tracker| tracker.summary().to_string())
    }

    /// Return `true` if all conservation violations are within acceptable limits.
    pub fn is_solution_valid(&self) -> bool {
        self.conservation_tracker
            .as_ref()
            .is_none_or(|tracker| tracker.is_solution_valid())
    }

    pub(super) fn check_stability(dt: f64, grid: &Grid, medium: &dyn Medium) -> bool {
        let max_c = medium
            .sound_speed_array()
            .iter()
            .fold(0.0_f64, |acc, &x| acc.max(x));
        let min_dx = grid.dx.min(grid.dy).min(grid.dz);
        let cfl = max_c * dt / min_dx;
        cfl < 0.5
    }

    /// Return disjoint pressure-history buffers for one leapfrog update.
    ///
    /// # Invariant
    /// `buffer_indices` is always a permutation of `[0, 1, 2]`, where entries
    /// mean `[next, current, previous]`. The returned references therefore
    /// satisfy the XOR ownership rule: exactly one mutable next buffer and two
    /// immutable history buffers. This keeps pressure history rotation
    /// allocation-free while preserving the three-level recurrence
    /// `p[n+1] = 2p\[n\] - p[n-1] + dt^2 rhs(p\[n\], p[n-1])`.
    pub(super) fn pressure_buffers_for_step(
        pressure_buffers: &mut [Array3<f64>; 3],
        buffer_indices: [usize; 3],
    ) -> (&mut Array3<f64>, &Array3<f64>, &Array3<f64>) {
        match buffer_indices {
            [0, 1, 2] => {
                let [next, current, previous] = pressure_buffers;
                (next, current, previous)
            }
            [0, 2, 1] => {
                let [next, previous, current] = pressure_buffers;
                (next, current, previous)
            }
            [1, 0, 2] => {
                let [current, next, previous] = pressure_buffers;
                (next, current, previous)
            }
            [1, 2, 0] => {
                let [previous, next, current] = pressure_buffers;
                (next, current, previous)
            }
            [2, 0, 1] => {
                let [current, previous, next] = pressure_buffers;
                (next, current, previous)
            }
            [2, 1, 0] => {
                let [previous, current, next] = pressure_buffers;
                (next, current, previous)
            }
            _ => unreachable!("buffer_indices must remain a permutation of [0, 1, 2]"),
        }
    }

    /// Get performance summary.
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[must_use]
    pub fn performance_summary(&self) -> String {
        let metrics = self.metrics.lock().unwrap();
        format!(
            "WesterveltWave Performance: {} calls, {:.2} ms total",
            metrics.call_count,
            metrics.total_time() * 1000.0
        )
    }

    pub(super) fn check_conservation_laws(&mut self) {
        let should_check = self.conservation_tracker.as_ref().is_some_and(|tracker| {
            self.current_step
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
            self.current_step,
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
                    warn!("Westervelt Wave Conservation Warning: {}", diag);
                }
                ViolationSeverity::Error => {
                    warn!("Westervelt Wave Conservation Error: {}", diag);
                }
                ViolationSeverity::Critical => {
                    warn!("Westervelt Wave Conservation CRITICAL: {}", diag);
                    warn!("   Solution may be physically invalid!");
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
