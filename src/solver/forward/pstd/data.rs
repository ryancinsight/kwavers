//! Spectral Solver Data Structures
//!
//! Internal data structures for the generalized spectral solver.
//! following GRASP principle of cohesive, focused modules.

use crate::domain::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::math::fft::Complex64;
use crate::domain::medium::Medium;
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
pub fn initialize_field_arrays(grid: &Grid, _medium: &dyn Medium) -> KwaversResult<FieldArrays> {
    let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);

    Ok(FieldArrays {
        p: Array3::zeros((nx, ny, nz)),
        p_k: Array3::zeros((nx, ny, nz)),
        ux: Array3::zeros((nx, ny, nz)),
        uy: Array3::zeros((nx, ny, nz)),
        uz: Array3::zeros((nx, ny, nz)),
    })
}
