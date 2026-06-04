//! `ViscousProperties` impl for `HeterogeneousTissueMedium`

use super::HeterogeneousTissueMedium;
use kwavers_grid::Grid;
use crate::medium::viscous::ViscousProperties;

impl ViscousProperties for HeterogeneousTissueMedium {
    fn viscosity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (i, j, k) = crate::medium::continuous_to_discrete(x, y, z, grid);
        self.get_tissue_properties(i, j, k).viscosity
    }

    fn shear_viscosity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        // Assume viscosity is shear viscosity for tissues
        self.viscosity(x, y, z, grid)
    }

    fn bulk_viscosity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        // Approximation for tissue
        self.viscosity(x, y, z, grid) * 2.5
    }
}
