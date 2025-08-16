//! Memory workspace for Kuznetsov equation solver

use crate::grid::Grid;
use crate::error::KwaversResult;
use ndarray::Array3;

/// RK4 time integration workspace
#[derive(Debug, Clone)]
pub struct RK4Workspace {
    /// Temporary arrays for RK4 stages
    k1: Array3<f64>,
    k2: Array3<f64>,
    k3: Array3<f64>,
    k4: Array3<f64>,
}

impl RK4Workspace {
    /// Create a new workspace for the given grid
    pub fn new(grid: &Grid) -> KwaversResult<Self> {
        let shape = (grid.nx, grid.ny, grid.nz);
        
        Ok(Self {
            k1: Array3::zeros(shape),
            k2: Array3::zeros(shape),
            k3: Array3::zeros(shape),
            k4: Array3::zeros(shape),
        })
    }
}