//! k-Wave Utility Functions
//!
//! This module contains helper functions for k-Wave implementation,
//! separated according to Single Responsibility Principle (GRASP).

use crate::grid::Grid;
use ndarray::Array3;

/// Compute PML absorption operators
pub(super) fn compute_pml_operators(grid: &Grid, pml_size: usize, pml_alpha: f64) -> Array3<f64> {
    let mut pml = Array3::ones((grid.nx, grid.ny, grid.nz));

    // Apply PML in each direction
    for ((i, j, k), pml_val) in pml.indexed_iter_mut() {
        let mut absorption = 0.0;

        // X boundaries
        if i < pml_size {
            let dist = (pml_size - i) as f64 / pml_size as f64;
            absorption += pml_alpha * dist * dist;
        } else if i >= grid.nx - pml_size {
            let dist = (i - (grid.nx - pml_size - 1)) as f64 / pml_size as f64;
            absorption += pml_alpha * dist * dist;
        }

        // Y boundaries (similar for other directions)
        if j < pml_size {
            let dist = (pml_size - j) as f64 / pml_size as f64;
            absorption += pml_alpha * dist * dist;
        } else if j >= grid.ny - pml_size {
            let dist = (j - (grid.ny - pml_size - 1)) as f64 / pml_size as f64;
            absorption += pml_alpha * dist * dist;
        }

        // Z boundaries
        if k < pml_size {
            let dist = (pml_size - k) as f64 / pml_size as f64;
            absorption += pml_alpha * dist * dist;
        } else if k >= grid.nz - pml_size {
            let dist = (k - (grid.nz - pml_size - 1)) as f64 / pml_size as f64;
            absorption += pml_alpha * dist * dist;
        }

        *pml_val = (-absorption).exp();
    }

    pml
}
