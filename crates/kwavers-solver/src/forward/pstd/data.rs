//! Spectral Solver Data Structures
//!
//! Internal data structures for the generalized spectral solver.
//! following GRASP principle of cohesive, focused modules.

use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_math::fft::Complex64;
use kwavers_medium::Medium;
use leto::Array3 as LetoArray3;

/// Helper struct for field array initialization
#[derive(Debug)]
pub struct FieldArrays {
    pub p: LetoArray3<f64>,
    pub p_k: LetoArray3<Complex64>,
    pub ux: LetoArray3<f64>,
    pub uy: LetoArray3<f64>,
    pub uz: LetoArray3<f64>,
}

/// Initialize field arrays for spectral solver
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub fn initialize_field_arrays(grid: &Grid, _medium: &dyn Medium) -> KwaversResult<FieldArrays> {
    let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
    let nz_c = nz / 2 + 1;

    Ok(FieldArrays {
        p: LetoArray3::zeros([nx, ny, nz]),
        // Half-spectrum buffer: r2c transform produces (nx, ny, nz/2+1) complex values.
        p_k: LetoArray3::zeros([nx, ny, nz_c]),
        ux: LetoArray3::zeros([nx, ny, nz]),
        uy: LetoArray3::zeros([nx, ny, nz]),
        uz: LetoArray3::zeros([nx, ny, nz]),
    })
}
