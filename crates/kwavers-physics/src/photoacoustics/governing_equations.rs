use super::thermoelasticity::GrueneisenModel;
use super::ThermoelasticReport;
use kwavers_core::error::KwaversResult;
use kwavers_imaging::photoacoustic::ThermoelasticProperties;

#[derive(Debug)]
pub struct PhotoacousticGoverningEquations;

impl PhotoacousticGoverningEquations {
    #[must_use]
    pub fn absorbed_energy_density(mu_a_m_inv: f64, fluence_j_m2: f64) -> f64 {
        mu_a_m_inv * fluence_j_m2
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
        let absorbed_energy_density_j_m3 = Self::absorbed_energy_density(mu_a_m_inv, fluence_j_m2);
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
