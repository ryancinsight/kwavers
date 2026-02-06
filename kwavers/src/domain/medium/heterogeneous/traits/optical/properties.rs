//! Optical properties implementation for heterogeneous media

use crate::domain::grid::Grid;
use crate::domain::medium::heterogeneous::{
    core::HeterogeneousMedium, interpolation::TrilinearInterpolator,
};
use crate::domain::medium::optical::OpticalProperties;

impl OpticalProperties for HeterogeneousMedium {
    /// Optical absorption coefficient μ_a at continuous coordinates (1/m)
    #[inline]
    fn optical_absorption_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        TrilinearInterpolator::get_field_value(
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
        TrilinearInterpolator::get_field_value(
            &self.mu_s_prime,
            x,
            y,
            z,
            grid,
            self.use_trilinear_interpolation,
        )
    }
}
