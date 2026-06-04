//! Viscous properties implementation for heterogeneous media

use kwavers_grid::Grid;
use crate::core::MIN_PHYSICAL_DENSITY;
use crate::heterogeneous::{
    core::HeterogeneousMedium, interpolation::HetTrilinearInterpolator,
};
use crate::viscous::ViscousProperties;

impl ViscousProperties for HeterogeneousMedium {
    /// Dynamic viscosity at continuous coordinates (Pa·s)
    #[inline]
    fn viscosity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        HetTrilinearInterpolator::get_field_value(
            &self.viscosity,
            x,
            y,
            z,
            grid,
            self.use_trilinear_interpolation,
        )
    }

    /// Shear viscosity (Pa·s) from coefficient field
    #[inline]
    fn shear_viscosity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        HetTrilinearInterpolator::get_field_value(
            &self.shear_viscosity_coeff,
            x,
            y,
            z,
            grid,
            self.use_trilinear_interpolation,
        )
    }

    /// Bulk viscosity (Pa·s) from coefficient field
    #[inline]
    fn bulk_viscosity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        HetTrilinearInterpolator::get_field_value(
            &self.bulk_viscosity_coeff,
            x,
            y,
            z,
            grid,
            self.use_trilinear_interpolation,
        )
    }

    /// Kinematic viscosity ν = μ / ρ (m²/s) using continuous interpolation
    #[inline]
    fn kinematic_viscosity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let mu = HetTrilinearInterpolator::get_field_value(
            &self.viscosity,
            x,
            y,
            z,
            grid,
            self.use_trilinear_interpolation,
        );
        let rho = HetTrilinearInterpolator::get_field_value(
            &self.density,
            x,
            y,
            z,
            grid,
            self.use_trilinear_interpolation,
        )
        .max(MIN_PHYSICAL_DENSITY);
        mu / rho
    }
}
