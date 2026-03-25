//! CPML memory variables for recursive convolution

use super::config::CPMLConfig;
use crate::domain::grid::Grid;
use ndarray::Array3;

/// CPML memory variables for field updates
/// Each field has 2 * thickness elements in its primary direction
/// to account for both left and right boundaries.
#[derive(Debug, Clone)]
pub struct CPMLMemory {
    // Memory for velocity updates (depends on pressure gradient)
    pub psi_p_x: Array3<f64>,
    pub psi_p_y: Array3<f64>,
    pub psi_p_z: Array3<f64>,

    // Memory for pressure updates (depends on velocity divergence)
    pub psi_v_x: Array3<f64>,
    pub psi_v_y: Array3<f64>,
    pub psi_v_z: Array3<f64>,
}

impl CPMLMemory {
    /// Create new memory variables with per-dimension PML thickness
    pub fn new(config: &CPMLConfig, grid: &Grid) -> Self {
        let tx = config.per_dimension.x;
        let ty = config.per_dimension.y;
        let tz = config.per_dimension.z;

        Self {
            psi_p_x: Array3::zeros((2 * tx, grid.ny, grid.nz)),
            psi_p_y: Array3::zeros((grid.nx, 2 * ty, grid.nz)),
            psi_p_z: Array3::zeros((grid.nx, grid.ny, 2 * tz)),

            psi_v_x: Array3::zeros((2 * tx, grid.ny, grid.nz)),
            psi_v_y: Array3::zeros((grid.nx, 2 * ty, grid.nz)),
            psi_v_z: Array3::zeros((grid.nx, grid.ny, 2 * tz)),
        }
    }

    /// Reset all memory variables to zero
    pub fn reset(&mut self) {
        self.psi_p_x.fill(0.0);
        self.psi_p_y.fill(0.0);
        self.psi_p_z.fill(0.0);
        self.psi_v_x.fill(0.0);
        self.psi_v_y.fill(0.0);
        self.psi_v_z.fill(0.0);
    }
}
