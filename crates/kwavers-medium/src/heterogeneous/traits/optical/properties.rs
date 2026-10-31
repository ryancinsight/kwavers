//! Optical properties implementation for heterogeneous media

use crate::heterogeneous::core::HeterogeneousMedium;
use crate::optical::{interaction_from_si, MediumOpticalProperties};
use aequitas::systems::si::quantities::ReciprocalLength;
use hyperion::{
    coefficient::{InteractionCoefficient, ReducedScattering},
    TransportError,
};
use kwavers_grid::Grid;

impl MediumOpticalProperties for HeterogeneousMedium {
    /// Optical absorption coefficient μ_a at continuous coordinates (1/m)
    #[inline]
    fn optical_absorption_coefficient(
        &self,
        x: f64,
        y: f64,
        z: f64,
        grid: &Grid,
    ) -> ReciprocalLength<f64> {
        ReciprocalLength::from_base(self.lookup_field(&self.mu_a, x, y, z, grid))
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
        interaction_from_si(self.lookup_field(&self.mu_s_prime, x, y, z, grid))
    }
}
