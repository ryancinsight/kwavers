//! Optical properties implementation for heterogeneous media

use crate::heterogeneous::{core::HeterogeneousMedium, interpolation::HetTrilinearInterpolator};
use crate::optical::{interaction_from_si, MediumOpticalProperties};
use hyperion::{
    coefficient::{InteractionCoefficient, ReducedScattering},
    TransportError,
};
use kwavers_grid::Grid;

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

    /// Reduced optical scattering coefficient μ_s' at continuous coordinates (1/m)
    #[inline]
    fn optical_reduced_scattering_coefficient(
        &self,
        x: f64,
        y: f64,
        z: f64,
        grid: &Grid,
    ) -> Result<InteractionCoefficient<f64, ReducedScattering>, TransportError<f64>> {
        interaction_from_si(HetTrilinearInterpolator::get_field_value(
            &self.mu_s_prime,
            x,
            y,
            z,
            grid,
            self.use_trilinear_interpolation,
        ))
    }
}
