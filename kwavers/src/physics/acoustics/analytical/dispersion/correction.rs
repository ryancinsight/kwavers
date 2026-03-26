//! Dispersion Correction Application
//!
//! Applies correction factors to 3D fields based on analytically
//! computed dispersion errors from FDTD or PSTD methods.

use super::DispersionAnalysis;
use crate::domain::grid::Grid;
use ndarray::Array3;
use std::f64::consts::PI;

/// Numerical method for dispersion calculation
#[derive(Debug, Clone, Copy)]
pub enum DispersionMethod {
    /// Finite-difference time-domain with timestep (1D analysis)
    FDTD(f64),
    /// Pseudo-spectral time-domain with order (1D analysis)
    PSTD(usize),
    /// Finite-difference time-domain with 3D analysis
    FDTD3D {
        /// Time step (s)
        dt: f64,
    },
    /// Pseudo-spectral time-domain with 3D analysis
    PSTD3D {
        /// Time step (s)
        dt: f64,
        /// Time-stepping order (2 or 4)
        order: usize,
    },
    /// No dispersion correction
    None,
}

impl DispersionAnalysis {
    /// Apply dispersion correction to a field (1D interface)
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
            DispersionMethod::PSTD(order) => {
                1.0 / (1.0 + Self::pstd_dispersion(k, grid.dx, order))
            }
            DispersionMethod::FDTD3D { .. } | DispersionMethod::PSTD3D { .. } => {
                eprintln!(
                    "Warning: Using 1D apply_correction with 3D method. \
                     Use apply_correction_3d for proper 3D dispersion handling."
                );
                return;
            }
            DispersionMethod::None => 1.0,
        };

        field.mapv_inplace(|v| v * correction_factor);
    }

    /// Apply dispersion correction to a field using full 3D analysis
    pub fn apply_correction_3d(
        field: &mut Array3<f64>,
        grid: &Grid,
        kx: f64,
        ky: f64,
        kz: f64,
        c: f64,
        method: DispersionMethod,
    ) {
        let correction_factor = match method {
            DispersionMethod::FDTD3D { dt } => {
                1.0 / (1.0
                    + Self::fdtd_dispersion_3d(kx, ky, kz, grid.dx, grid.dy, grid.dz, dt, c))
            }
            DispersionMethod::PSTD3D { dt, order } => {
                1.0 / (1.0
                    + Self::pstd_dispersion_3d(
                        kx, ky, kz, grid.dx, grid.dy, grid.dz, dt, c, order,
                    ))
            }
            DispersionMethod::FDTD(dt) => {
                let k_magnitude = (kx * kx + ky * ky + kz * kz).sqrt();
                1.0 / (1.0 + Self::fdtd_dispersion(k_magnitude, grid.dx, dt, c))
            }
            DispersionMethod::PSTD(order) => {
                let k_magnitude = (kx * kx + ky * ky + kz * kz).sqrt();
                1.0 / (1.0 + Self::pstd_dispersion(k_magnitude, grid.dx, order))
            }
            DispersionMethod::None => 1.0,
        };

        field.mapv_inplace(|v| v * correction_factor);
    }
}
