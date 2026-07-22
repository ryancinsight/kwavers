//! Optical properties trait for light propagation
//!
//! This module defines traits for optical properties including absorption,
//! scattering, and refractive index.

use crate::core::CoreMedium;
use aequitas::systems::si::{quantities::ReciprocalLength, units::PerMeter};
use hyperion::{
    coefficient::{CoefficientRole, InteractionCoefficient, ReducedScattering},
    TransportError,
};
use kwavers_grid::Grid;

pub(crate) fn interaction_from_si<Role: CoefficientRole>(
    value_per_meter: f64,
) -> Result<InteractionCoefficient<f64, Role>, TransportError<f64>> {
    InteractionCoefficient::new(ReciprocalLength::from_unit::<PerMeter>(value_per_meter))
}

/// Trait for optical medium properties
pub trait MediumOpticalProperties: CoreMedium {
    /// Get optical absorption coefficient (1/m).
    fn optical_absorption_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;

    /// Get the reduced scattering coefficient mu_s' (1/m).
    ///
    /// # Errors
    ///
    /// Returns a Hyperion transport error when the stored coefficient is
    /// negative or non-finite.
    fn optical_reduced_scattering_coefficient(
        &self,
        x: f64,
        y: f64,
        z: f64,
        grid: &Grid,
    ) -> Result<InteractionCoefficient<f64, ReducedScattering>, TransportError<f64>>;
}

#[cfg(test)]
mod tests {
    use super::interaction_from_si;
    use hyperion::{coefficient::ReducedScattering, TransportError, ValueConstraint, ValueKind};

    #[test]
    fn reduced_scattering_uses_hyperion_validation() {
        let expected = 22.4;
        let reduced = interaction_from_si::<ReducedScattering>(expected).unwrap();
        assert!(
            (reduced.in_unit::<aequitas::systems::si::units::PerMeter>() - expected).abs()
                <= 16.0 * f64::EPSILON * expected
        );
    }

    #[test]
    fn reduced_scattering_rejects_invalid_inputs() {
        assert_eq!(
            interaction_from_si::<ReducedScattering>(-1.0),
            Err(TransportError::InvalidValue {
                field: ValueKind::ReducedScatteringCoefficient,
                value: -1.0,
                constraint: ValueConstraint::FiniteNonNegative,
            })
        );
        match interaction_from_si::<ReducedScattering>(f64::NAN) {
            Err(TransportError::InvalidValue {
                field: ValueKind::ReducedScatteringCoefficient,
                value,
                constraint: ValueConstraint::FiniteNonNegative,
            }) => assert!(value.is_nan()),
            result => panic!("expected a non-finite coefficient error, got {result:?}"),
        }
    }
}
