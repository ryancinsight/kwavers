use super::{GrueneisenModel, ThermoelasticReport};
use aequitas::systems::si::{
    quantities::{EnergyPerArea, ReciprocalLength},
    units::{JoulePerCubicMeter, JoulePerSquareMeter, PerMeter},
};
use hyperion::{
    coefficient::{Absorption, InteractionCoefficient},
    quantity::EnergyFluence,
    transport, TransportError,
};
use kwavers_core::error::KwaversResult;
use kwavers_core::error::{KwaversError, ValidationError};
use kwavers_imaging::photoacoustic::ThermoelasticProperties;

#[derive(Debug)]
pub struct PhotoacousticGoverningEquations;

impl PhotoacousticGoverningEquations {
    pub fn absorbed_energy_density(mu_a_m_inv: f64, fluence_j_m2: f64) -> KwaversResult<f64> {
        let absorption = InteractionCoefficient::<f64, Absorption>::new(
            ReciprocalLength::from_unit::<PerMeter>(mu_a_m_inv),
        )
        .map_err(map_transport_error)?;
        let fluence = EnergyFluence::new(EnergyPerArea::from_unit::<JoulePerSquareMeter>(
            fluence_j_m2,
        ))
        .map_err(map_transport_error)?;
        let absorbed =
            transport::absorbed_energy_density(absorption, fluence).map_err(map_transport_error)?;
        Ok(absorbed.in_unit::<JoulePerCubicMeter>())
    }

    /// Compute initial pressure using a [`GrueneisenModel`] at the given temperature.
    ///
    /// ## Formula
    ///
    /// ```text
    /// p₀ = Γ(T) · μ_a · Φ
    /// ```
    ///
    /// Callers that only have a bare `f64` Grüneisen coefficient should construct
    /// `GrueneisenModel::constant(gamma)` and pass body temperature (37.0 °C).
    /// # Errors
    /// - Propagates invalid confinement-domain parameters.
    pub fn initial_pressure(
        mu_a_m_inv: f64,
        fluence_j_m2: f64,
        pulse_duration_s: f64,
        thermoelastic: ThermoelasticProperties,
        gruneisen: &GrueneisenModel,
        temperature_celsius: f64,
    ) -> KwaversResult<ThermoelasticReport> {
        let absorbed_energy_density_j_m3 = Self::absorbed_energy_density(mu_a_m_inv, fluence_j_m2)?;
        ThermoelasticReport::from_absorbed_energy(
            absorbed_energy_density_j_m3,
            mu_a_m_inv,
            pulse_duration_s,
            thermoelastic,
            gruneisen,
            temperature_celsius,
        )
    }
}

fn map_transport_error(error: TransportError<f64>) -> KwaversError {
    KwaversError::Validation(ValidationError::InvalidParameter {
        parameter: "photoacoustic_optics".to_owned(),
        reason: error.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::PhotoacousticGoverningEquations;

    #[test]
    fn absorbed_energy_density_routes_through_hyperion_transport() {
        let absorbed = PhotoacousticGoverningEquations::absorbed_energy_density(120.0, 3.5)
            .expect("valid optical parameters");
        assert!((absorbed - 420.0).abs() <= 8.0 * f64::EPSILON * 420.0);
    }

    #[test]
    fn absorbed_energy_density_rejects_invalid_optical_inputs() {
        let error = PhotoacousticGoverningEquations::absorbed_energy_density(-1.0, 3.5)
            .expect_err("negative absorption must be rejected");
        assert!(format!("{error}").contains("photoacoustic_optics"));
    }
}
