use crate::domain::imaging::photoacoustic::{PhotoacousticScenario, ThermoelasticProperties};

/// Validity report for retained photoacoustic physics assumptions.
#[derive(Debug, Clone, Copy)]
pub struct PhotoacousticValidityReport {
    pub diffusion_regime_valid: bool,
    pub stress_confined: bool,
    pub thermal_confined: bool,
}

impl PhotoacousticValidityReport {
    #[must_use]
    pub fn evaluate(
        scenario: &PhotoacousticScenario,
        mu_a: f64,
        mu_s_prime: f64,
        thermoelastic: ThermoelasticProperties,
    ) -> Self {
        let confinement = super::ConfinementAssessment::evaluate(
            mu_a,
            scenario.config.pulse_duration_s,
            thermoelastic,
        );
        Self {
            diffusion_regime_valid: mu_s_prime > mu_a,
            stress_confined: confinement.stress_confined,
            thermal_confined: confinement.thermal_confined,
        }
    }
}
