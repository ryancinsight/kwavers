//! Dispersion Correction Application
//!
//! Applies correction factors to 3D fields based on analytically
//! computed dispersion errors from FDTD or PSTD methods.

use super::DispersionAnalysis;
use crate::core::constants::numerical::TWO_PI;
use crate::domain::grid::Grid;
use ndarray::Array3;

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
        let k = TWO_PI * frequency / c;

        let correction_factor = match method {
            DispersionMethod::FDTD(dt) => 1.0 / (1.0 + Self::fdtd_dispersion(k, grid.dx, dt, c)),
            DispersionMethod::PSTD(order) => 1.0 / (1.0 + Self::pstd_dispersion(k, grid.dx, order)),
            DispersionMethod::FDTD3D { .. } | DispersionMethod::PSTD3D { .. } => {
                eprintln!(
                    "Warning: Using 1D apply_correction with 3D method. \
                     Use apply_correction_3d for proper 3D dispersion handling."
                );
                return;
            }
            DispersionMethod::None => 1.0,
        };

        field.par_mapv_inplace(|v| v * correction_factor);
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
                1.0 / (1.0 + Self::fdtd_dispersion_3d(kx, ky, kz, grid.dx, grid.dy, grid.dz, dt, c))
            }
            DispersionMethod::PSTD3D { dt, order } => {
                1.0 / (1.0
                    + Self::pstd_dispersion_3d(kx, ky, kz, grid.dx, grid.dy, grid.dz, dt, c, order))
            }
            DispersionMethod::FDTD(dt) => {
                let k_magnitude = kz.mul_add(kz, kx.mul_add(kx, ky * ky)).sqrt();
                1.0 / (1.0 + Self::fdtd_dispersion(k_magnitude, grid.dx, dt, c))
            }
            DispersionMethod::PSTD(order) => {
                let k_magnitude = kz.mul_add(kz, kx.mul_add(kx, ky * ky)).sqrt();
                1.0 / (1.0 + Self::pstd_dispersion(k_magnitude, grid.dx, order))
            }
            DispersionMethod::None => 1.0,
        };

        field.par_mapv_inplace(|v| v * correction_factor);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;
    use crate::core::constants::numerical::MHZ_TO_HZ;

    fn unit_grid() -> Grid {
        Grid::new(8, 8, 8, 1e-4, 1e-4, 1e-4).expect("grid must be created")
    }

    /// DispersionMethod::None leaves the field unchanged.
    #[test]
    fn none_method_leaves_field_unchanged() {
        let grid = unit_grid();
        let mut field = Array3::from_elem((8, 8, 8), 2.0_f64);
        DispersionAnalysis::apply_correction(
            &mut field,
            &grid,
            MHZ_TO_HZ,
            SOUND_SPEED_WATER_SIM,
            DispersionMethod::None,
        );
        for &v in field.iter() {
            assert!(
                (v - 2.0).abs() < 1e-14,
                "None method must not change field (got {v})"
            );
        }
    }

    /// DispersionMethod::PSTD(2) applies a strictly positive correction factor.
    ///
    /// For valid k·dx ≪ 1, ε = 0.02·(k·dx)² ≥ 0 → correction factor 1/(1+ε) ≤ 1.
    #[test]
    fn pstd2_correction_reduces_field_amplitude() {
        let grid = unit_grid();
        let mut field = Array3::from_elem((8, 8, 8), 1.0_f64);
        // 1 MHz in water at 8 PPW → correction reduces amplitude
        DispersionAnalysis::apply_correction(
            &mut field,
            &grid,
            MHZ_TO_HZ,
            SOUND_SPEED_WATER_SIM,
            DispersionMethod::PSTD(2),
        );
        for &v in field.iter() {
            assert!(
                v > 0.0 && v <= 1.0 + 1e-10,
                "PSTD(2) correction must give values in (0,1]: got {v}"
            );
        }
    }

    /// apply_correction_3d with None leaves field unchanged.
    #[test]
    fn none_method_3d_leaves_field_unchanged() {
        let grid = unit_grid();
        let mut field = Array3::from_elem((8, 8, 8), 3.0_f64);
        DispersionAnalysis::apply_correction_3d(
            &mut field,
            &grid,
            0.0,
            0.0,
            0.0,
            SOUND_SPEED_WATER_SIM,
            DispersionMethod::None,
        );
        for &v in field.iter() {
            assert!(
                (v - 3.0).abs() < 1e-14,
                "None method must not change field (got {v})"
            );
        }
    }
}
