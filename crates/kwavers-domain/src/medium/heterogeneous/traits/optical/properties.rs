//! Optical properties implementation for heterogeneous media

use kwavers_grid::Grid;
use crate::medium::heterogeneous::{
    core::HeterogeneousMedium, interpolation::HetTrilinearInterpolator,
};
use crate::medium::optical::MediumOpticalProperties;

impl MediumOpticalProperties for HeterogeneousMedium {
    /// Optical absorption coefficient μ_a at continuous coordinates (1/m)
    #[inline]
    fn optical_absorption_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        HetTrilinearInterpolator::get_field_value(
            &self.mu_a,
            x,
            y,
            z,
            grid,
            self.use_trilinear_interpolation,
        )
    }

    /// Optical scattering coefficient μ_s at continuous coordinates (1/m)
    #[inline]
    fn optical_scattering_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        HetTrilinearInterpolator::get_field_value(
            &self.mu_s_prime,
            x,
            y,
            z,
            grid,
            self.use_trilinear_interpolation,
        )
    }
}
