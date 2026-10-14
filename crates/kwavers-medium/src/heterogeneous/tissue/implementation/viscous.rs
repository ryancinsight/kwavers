//! `ViscousProperties` impl for `HeterogeneousTissueMedium`

use super::HeterogeneousTissueMedium;
use crate::viscous::ViscousProperties;
use aequitas::systems::si::quantities::DynamicViscosity;
use kwavers_grid::Grid;

impl ViscousProperties for HeterogeneousTissueMedium {
    fn viscosity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> DynamicViscosity {
        let (i, j, k) = crate::continuous_to_discrete(x, y, z, grid);
        DynamicViscosity::from_base(self.get_tissue_properties(i, j, k).viscosity)
    }

    fn shear_viscosity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> DynamicViscosity {
        // Assume viscosity is shear viscosity for tissues
        self.viscosity(x, y, z, grid)
    }

    fn bulk_viscosity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> DynamicViscosity {
        // Approximation for tissue
        DynamicViscosity::from_base(self.viscosity(x, y, z, grid).into_base() * 2.5)
    }
}
