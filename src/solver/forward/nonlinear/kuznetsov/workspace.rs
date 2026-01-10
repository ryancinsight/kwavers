//! Workspace management for Kuznetsov solver
//!
//! This module provides pre-allocated workspace arrays to eliminate
//! heap allocations in the main simulation loop.

use super::spectral::SpectralOperator;
use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use ndarray::Array3;

/// Comprehensive workspace for Kuznetsov equation solver
///
/// Pre-allocates all temporary arrays needed for computation to avoid
/// allocations in the hot loop.
#[derive(Debug)]
pub struct KuznetsovWorkspace {
    /// Spectral operator for FFT-based derivatives
    pub spectral_op: SpectralOperator,

    /// Pressure field at previous time steps (for finite differences)
    pub pressure_prev: Array3<f64>,
    pub pressure_prev2: Array3<f64>,
    pub pressure_prev3: Array3<f64>,

    /// Workspace for nonlinear term computation
    pub nonlinear_term: Array3<f64>,

    /// Workspace for diffusive term computation
    pub diffusive_term: Array3<f64>,

    /// Workspace for Laplacian computation
    pub laplacian: Array3<f64>,

    /// Workspace for gradient computation
    pub grad_x: Array3<f64>,
    pub grad_y: Array3<f64>,
    pub grad_z: Array3<f64>,

    /// RK4 intermediate stages
    pub k1: Array3<f64>,
    pub k2: Array3<f64>,
    pub k3: Array3<f64>,
    pub k4: Array3<f64>,

    /// Temporary field for RK4 stages
    pub temp_field: Array3<f64>,
}

impl KuznetsovWorkspace {
    /// Create a new workspace for the given grid
    pub fn new(grid: &Grid) -> KwaversResult<Self> {
        let shape = (grid.nx, grid.ny, grid.nz);

        Ok(Self {
            spectral_op: SpectralOperator::new(grid),

            // Time history buffers
            pressure_prev: Array3::zeros(shape),
            pressure_prev2: Array3::zeros(shape),
            pressure_prev3: Array3::zeros(shape),

            // Term computation buffers
            nonlinear_term: Array3::zeros(shape),
            diffusive_term: Array3::zeros(shape),
            laplacian: Array3::zeros(shape),

            // Gradient buffers
            grad_x: Array3::zeros(shape),
            grad_y: Array3::zeros(shape),
            grad_z: Array3::zeros(shape),

            // RK4 buffers
            k1: Array3::zeros(shape),
            k2: Array3::zeros(shape),
            k3: Array3::zeros(shape),
            k4: Array3::zeros(shape),
            temp_field: Array3::zeros(shape),
        })
    }

    /// Update time history buffers
    pub fn update_time_history(&mut self, current_pressure: &Array3<f64>) {
        // Shift history: prev3 <- prev2 <- prev <- current
        self.pressure_prev3.assign(&self.pressure_prev2);
        self.pressure_prev2.assign(&self.pressure_prev);
        self.pressure_prev.assign(current_pressure);
    }
}
