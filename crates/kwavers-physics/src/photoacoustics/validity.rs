use kwavers_core::error::KwaversResult;
use kwavers_domain::imaging::photoacoustic::{PhotoacousticScenario, ThermoelasticProperties};

/// Validity report for retained photoacoustic physics assumptions.
#[derive(Debug, Clone, Copy)]
pub struct PhotoacousticValidityReport {
    pub diffusion_regime_valid: bool,
    pub stress_confined: bool,
    pub thermal_confined: bool,
}

impl PhotoacousticValidityReport {
    /// Evaluate retained photoacoustic validity assumptions.
    ///
    /// `diffusion_regime_valid` requires the diffusion approximation
    /// criterion `μ_s' ≥ 10 · μ_a` (Jacques 2013, *Phys. Med. Biol.* 58
    /// (11), R37 §3.1; standard P1 / Eddington diffusion-theory limit).
    /// Prior to 2026-05-21 this used `μ_s' > μ_a` — a far weaker test
    /// that flagged the diffusion regime as valid for marginal cases
    /// (e.g. `μ_s' = 1.01 · μ_a`) where the radiative-transfer equation
    /// cannot be approximated by a P1 expansion.
    ///
    /// # Errors
    /// - Propagates invalid confinement-domain parameters.
    pub fn evaluate(
        scenario: &PhotoacousticScenario,
        mu_a: f64,
        mu_s_prime: f64,
        thermoelastic: ThermoelasticProperties,
    ) -> KwaversResult<Self> {
        let confinement = super::ConfinementAssessment::evaluate(
            mu_a,
            scenario.config.pulse_duration_s,
            thermoelastic,
        )?;
        Ok(Self {
            // Jacques 2013 diffusion-regime criterion: μ_s' >> μ_a (factor ≥10).
            diffusion_regime_valid: mu_a > 0.0 && mu_s_prime >= 10.0 * mu_a,
            stress_confined: confinement.stress_confined,
            thermal_confined: confinement.thermal_confined,
        })
    }
}
