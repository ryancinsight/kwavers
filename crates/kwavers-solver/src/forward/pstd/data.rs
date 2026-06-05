//! Spectral Solver Data Structures
//!
//! Internal data structures for the generalized spectral solver.
//! following GRASP principle of cohesive, focused modules.

use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_math::fft::Complex64;
use kwavers_medium::Medium;
use ndarray::Array3;

/// Helper struct for field array initialization
#[derive(Debug)]
pub struct FieldArrays {
    pub p: Array3<f64>,
    pub p_k: Array3<Complex64>,
    pub ux: Array3<f64>,
    pub uy: Array3<f64>,
    pub uz: Array3<f64>,
}

/// Initialize field arrays for spectral solver
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub fn initialize_field_arrays(grid: &Grid, _medium: &dyn Medium) -> KwaversResult<FieldArrays> {
    let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
    let nz_c = nz / 2 + 1;

    Ok(FieldArrays {
        p: Array3::zeros((nx, ny, nz)),
        // Half-spectrum buffer: r2c transform produces (nx, ny, nz/2+1) complex values.
        p_k: Array3::zeros((nx, ny, nz_c)),
        ux: Array3::zeros((nx, ny, nz)),
        uy: Array3::zeros((nx, ny, nz)),
        uz: Array3::zeros((nx, ny, nz)),
    })
}
