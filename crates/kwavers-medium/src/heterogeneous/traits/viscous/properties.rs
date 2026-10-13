//! Viscous properties implementation for heterogeneous media

use crate::core::MIN_PHYSICAL_DENSITY;
use crate::heterogeneous::core::HeterogeneousMedium;
use crate::viscous::ViscousProperties;
use kwavers_grid::Grid;

impl ViscousProperties for HeterogeneousMedium {
    /// Dynamic viscosity at continuous coordinates (Pa·s)
    #[inline]
    fn viscosity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        self.lookup_field(&self.viscosity, x, y, z, grid)
    }

    /// Shear viscosity (Pa·s) from coefficient field
    #[inline]
    fn shear_viscosity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        self.lookup_field(&self.shear_viscosity_coeff, x, y, z, grid)
    }

    /// Bulk viscosity (Pa·s) from coefficient field
    #[inline]
    fn bulk_viscosity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        self.lookup_field(&self.bulk_viscosity_coeff, x, y, z, grid)
    }

    /// Kinematic viscosity ν = μ / ρ (m²/s) using continuous interpolation
    #[inline]
    fn kinematic_viscosity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let mu = self.lookup_field(&self.viscosity, x, y, z, grid);
        let rho = self
            .lookup_field(&self.density, x, y, z, grid)
            .max(MIN_PHYSICAL_DENSITY);
        mu / rho
    }
}
