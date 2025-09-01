//! CPML memory variables for recursive convolution

use super::config::CPMLConfig;
use crate::grid::Grid;
use ndarray::Array3;

/// CPML memory variables for field updates
#[derive(Debug, Clone)]
pub struct CPMLMemory {
    pub psi_vx_x: Array3<f64>,
    pub psi_vy_y: Array3<f64>,
    pub psi_vz_z: Array3<f64>,
    pub psi_p_x: Array3<f64>,
    pub psi_p_y: Array3<f64>,
    pub psi_p_z: Array3<f64>,
}

impl CPMLMemory {
    /// Create new memory variables
    pub fn new(config: &CPMLConfig, grid: &Grid) -> Self {
        let thickness = config.thickness;

        Self {
            psi_vx_x: Array3::zeros((thickness, grid.ny, grid.nz)),
            psi_vy_y: Array3::zeros((grid.nx, thickness, grid.nz)),
            psi_vz_z: Array3::zeros((grid.nx, grid.ny, thickness)),
            psi_p_x: Array3::zeros((thickness, grid.ny, grid.nz)),
            psi_p_y: Array3::zeros((grid.nx, thickness, grid.nz)),
            psi_p_z: Array3::zeros((grid.nx, grid.ny, thickness)),
        }
    }

    /// Reset all memory variables to zero
    pub fn reset(&mut self) {
        self.psi_vx_x.fill(0.0);
        self.psi_vy_y.fill(0.0);
        self.psi_vz_z.fill(0.0);
        self.psi_p_x.fill(0.0);
        self.psi_p_y.fill(0.0);
        self.psi_p_z.fill(0.0);
    }
}
