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
//! where `|k|² = kx² + ky² + kz²` is the squared wavenumber magnitude stored
//! in `k_squared`. This gives spectral accuracy (exponential convergence) for all
//! smooth solutions, outperforming 2nd-order FD which converges as O(Δx²).
//!
//! ## Time-Stepping (Leapfrog)
//!
//! A 3-level leapfrog scheme is used for the second-order time derivative:
//! ```text
//!   p^{n+1} = 2p^n − p^{n−1} + (c₀Δt)² · ∇²p^n
//!           + (δΔt/c₀⁴) · (p^n − p^{n−1}) / Δt      [thermoviscous]
//!           + (βΔt²/ρ₀c₀⁴) · ∂²(p²)/∂t²              [nonlinear]
//! ```
//! Stability requires the acoustic CFL condition: `c₀ · Δt · |k|_max ≤ 2`.
//!
//! ## References
//! - Westervelt, P.J. (1963). Parametric acoustic array. J. Acoust. Soc. Am. 35(4), 535–537.
//! - Hamilton, M.F. & Blackstock, D.T. (1998). Nonlinear Acoustics. Academic Press, Ch. 3.

use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use crate::math::fft::Complex64;
use crate::solver::forward::nonlinear::conservation::{
    ConservationDiagnostics, ConservationTolerances, ConservationTracker, ViolationSeverity,
};
use log::warn;
use ndarray::Array3;
use std::sync::{Arc, Mutex};

use super::metrics::PerformanceMetrics;
use super::spectral::initialize_kspace_grids;

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
    pub(super) k_squared: Option<Array3<f64>>,
    pub(super) nonlinearity_scaling: f64,
    /// Pressure history using buffer rotation for zero-allocation updates.
    pub(super) pressure_buffers: [Array3<f64>; 3],
    pub(super) buffer_indices: [usize; 3], // [next, current, previous]
    /// Pre-allocated complex scratch for spectral Laplacian FFT.
    pub(super) fft_scratch: Option<Array3<Complex64>>,
    /// Pre-allocated real scratch for spectral Laplacian IFFT.
    pub(super) laplacian_scratch: Option<Array3<f64>>,
    pub(super) metrics: Arc<Mutex<PerformanceMetrics>>,
    pub(super) conservation_tracker: Option<ConservationTracker>,
    pub(super) current_step: usize,
    pub(super) current_time: f64,
    pub(super) grid_cache: Option<Grid>,
    pub(super) medium_properties: Option<MediumProperties>,
}

impl WesterveltWave {
    /// Create a new Westervelt solver.
    pub fn new(grid: &Grid) -> Self {
        let (k_squared, _kx, _ky, _kz) = initialize_kspace_grids(grid);
        let shape = (grid.nx, grid.ny, grid.nz);

        let fft_scratch = Some(Array3::<Complex64>::zeros(shape));
        let laplacian_scratch = Some(Array3::<f64>::zeros(shape));

        Self {
            k_squared: Some(k_squared),
            nonlinearity_scaling: 1.0,
            pressure_buffers: [
                Array3::zeros(shape),
                Array3::zeros(shape),
                Array3::zeros(shape),
            ],
            buffer_indices: [0, 1, 2],
            fft_scratch,
            laplacian_scratch,
            metrics: Arc::new(Mutex::new(PerformanceMetrics::new())),
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
            let center_x = grid.dx * (grid.nx as f64) / 2.0;
            let center_y = grid.dy * (grid.ny as f64) / 2.0;
            let center_z = grid.dz * (grid.nz as f64) / 2.0;
            let rho0 =
                crate::domain::medium::density_at(medium, center_x, center_y, center_z, grid);
            let c0 =
                crate::domain::medium::sound_speed_at(medium, center_x, center_y, center_z, grid);
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

    pub(super) fn check_stability(
        &self,
        dt: f64,
        grid: &Grid,
        medium: &dyn Medium,
        _pressure: &Array3<f64>,
    ) -> bool {
        let max_c = medium
            .sound_speed_array()
            .iter()
            .fold(0.0_f64, |acc, &x| acc.max(x));
        let min_dx = grid.dx.min(grid.dy).min(grid.dz);
        let cfl = max_c * dt / min_dx;
        cfl < 0.5
    }

    pub(super) fn initialize_buffers(&mut self, initial_pressure: &Array3<f64>) {
        self.pressure_buffers[self.buffer_indices[1]].assign(initial_pressure);
        self.pressure_buffers[self.buffer_indices[2]].assign(initial_pressure);
    }

    /// Get performance summary.
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

/// Compute Laplacian using second-order finite differences (fallback).
pub(super) fn compute_laplacian_fd(field: &Array3<f64>, grid: &Grid) -> Array3<f64> {
    let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
    let mut laplacian = Array3::zeros((nx, ny, nz));

    let dx2_inv = 1.0 / (grid.dx * grid.dx);
    let dy2_inv = 1.0 / (grid.dy * grid.dy);
    let dz2_inv = 1.0 / (grid.dz * grid.dz);

    for k in 1..nz - 1 {
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                laplacian[[i, j, k]] = (field[[i + 1, j, k]] - 2.0 * field[[i, j, k]]
                    + field[[i - 1, j, k]])
                    * dx2_inv
                    + (field[[i, j + 1, k]] - 2.0 * field[[i, j, k]] + field[[i, j - 1, k]])
                        * dy2_inv
                    + (field[[i, j, k + 1]] - 2.0 * field[[i, j, k]] + field[[i, j, k - 1]])
                        * dz2_inv;
            }
        }
    }

    laplacian
}
