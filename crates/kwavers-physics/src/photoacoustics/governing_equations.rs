use super::{GrueneisenModel, ThermoelasticReport};
use aequitas::systems::si::{
    quantities::{EnergyPerArea, ReciprocalLength},
    units::{JoulePerCubicMeter, JoulePerSquareMeter, PerMeter},
};
use hyperion::{
    coefficient::{Absorption, InteractionCoefficient},
    quantity::EnergyFluence,
    TransportError,
};
use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};
use kwavers_imaging::photoacoustic::ThermoelasticProperties;

#[derive(Debug)]
pub struct PhotoacousticGoverningEquations;

impl PhotoacousticGoverningEquations {
    /// Absorbed energy density `q = μ_a Φ`, in `J/m³`.
    ///
    /// The product itself is Hyperion's law; this wrapper only adapts the
    /// coherent-SI scalars this module works in.
    ///
    /// # Errors
    /// - Returns [`Err`] when Hyperion rejects a negative or non-finite
    ///   absorption coefficient or fluence.
    pub fn absorbed_energy_density(mu_a_m_inv: f64, fluence_j_m2: f64) -> KwaversResult<f64> {
        let absorption = InteractionCoefficient::<f64, Absorption>::new(
            ReciprocalLength::from_unit::<PerMeter>(mu_a_m_inv),
        )
        .map_err(map_transport_error)?;
        let incident = EnergyFluence::new(EnergyPerArea::from_unit::<JoulePerSquareMeter>(
            fluence_j_m2,
        ))
        .map_err(map_transport_error)?;

        Ok(
            hyperion::transport::absorbed_energy_density(absorption, incident)
                .map_err(map_transport_error)?
                .in_unit::<JoulePerCubicMeter>(),
        )
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
    KwaversError::Validation(ValidationError::ConstraintViolation {
        message: format!("Hyperion optical transport rejected photoacoustic input: {error}"),
    })
}
