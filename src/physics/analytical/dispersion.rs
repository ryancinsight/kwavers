//! Dispersion analysis and correction for numerical methods

use crate::grid::Grid;
use ndarray::Array3;
use std::f64::consts::PI;

/// Dispersion analysis for numerical methods
#[derive(Debug)]
pub struct DispersionAnalysis;

impl DispersionAnalysis {
    /// Calculate numerical dispersion for FDTD method
    #[must_use]
    pub fn fdtd_dispersion(k: f64, dx: f64, dt: f64, c: f64) -> f64 {
        let cfl = c * dt / dx;
        let kx_dx = k * dx;

        // Von Neumann stability analysis result
        let sin_half_omega_dt = (cfl * kx_dx.sin()).asin();
        let omega_numerical = 2.0 * sin_half_omega_dt / dt;
        let omega_exact = k * c;

        (omega_numerical - omega_exact) / omega_exact
    }

    /// Calculate numerical dispersion for PSTD method
    #[must_use]
    pub fn pstd_dispersion(k: f64, dx: f64, order: usize) -> f64 {
        // K-space method dispersion (spectral accuracy)
        let kx_dx = k * dx;

        match order {
            2 => 0.02 * kx_dx.powi(2),  // Second-order correction
            4 => 0.001 * kx_dx.powi(4), // Fourth-order correction
            _ => 0.0,                   // Perfect for lower orders
        }
    }

    /// Apply dispersion correction to a field
    pub fn apply_correction(
        field: &mut Array3<f64>,
        grid: &Grid,
        frequency: f64,
        c: f64,
        method: DispersionMethod,
    ) {
        let k = 2.0 * PI * frequency / c;

        let correction_factor = match method {
            DispersionMethod::FDTD(dt) => 1.0 / (1.0 + Self::fdtd_dispersion(k, grid.dx, dt, c)),
            DispersionMethod::PSTD(order) => 1.0 / (1.0 + Self::pstd_dispersion(k, grid.dx, order)),
            DispersionMethod::None => 1.0,
        };

        field.mapv_inplace(|v| v * correction_factor);
    }
}

/// Numerical method for dispersion calculation
#[derive(Debug)]
pub enum DispersionMethod {
    /// Finite-difference time-domain with timestep
    FDTD(f64),
    /// Pseudo-spectral time-domain with order
    PSTD(usize),
    /// No dispersion correction
    None,
}
