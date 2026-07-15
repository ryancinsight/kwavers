use super::{ConfinementAssessment, GrueneisenModel};
use kwavers_core::error::KwaversResult;
use kwavers_imaging::photoacoustic::ThermoelasticProperties;

#[derive(Debug, Clone, Copy)]
pub struct ThermoelasticReport {
    pub absorbed_energy_density_j_m3: f64,
    pub initial_pressure_pa: f64,
    pub confinement: ConfinementAssessment,
}

impl ThermoelasticReport {
    /// Compute the thermoelastic report using a [`GrueneisenModel`] at a specific
    /// tissue temperature.
    ///
    /// ## Theorem (Sigrist 1986; Xu & Wang 2006, Eq. 2)
    ///
    /// Initial photoacoustic pressure:
    ///
    /// ```text
    /// p₀ = Γ(T) · μ_a · Φ = Γ(T) · H
    /// ```
    ///
    /// where `H = μ_a · Φ` is the absorbed energy density \[J/m³\] and
    /// `Γ(T) = Γ₀ + c_T · (T − T_ref)` is evaluated at the local tissue temperature.
    ///
    /// For a temperature-independent medium (constant Γ), construct the model with
    /// `GrueneisenModel::constant(gamma_0)`.  For tissue-specific defaults use
    /// `GrueneisenModel::soft_tissue()` or `GrueneisenModel::water()`.
    ///
    /// **This is the sole path** — the legacy temperature-independent overload has been
    /// removed to enforce the Grüneisen SSOT (Sprint 226).  Callers that formerly
    /// passed a bare `f64` coefficient should construct
    /// `GrueneisenModel::constant(gamma_value)` and pass `37.0` for body temperature.
    ///
    /// ## References
    /// - Sigrist MW (1986). *J Appl Phys* **60**(7), R83. DOI: 10.1063/1.337089
    /// - Xu M, Wang LV (2006). *Rev Sci Instrum* **77**, 041101. DOI: 10.1063/1.2195024
    ///
    /// # Errors
    /// - Propagates invalid confinement-domain parameters.
    pub fn from_absorbed_energy(
        absorbed_energy_density_j_m3: f64,
        mu_a_m_inv: f64,
        pulse_duration_s: f64,
        thermoelastic: ThermoelasticProperties,
        model: &GrueneisenModel,
        temperature_celsius: f64,
    ) -> KwaversResult<Self> {
        let confinement =
            ConfinementAssessment::evaluate(mu_a_m_inv, pulse_duration_s, thermoelastic)?;
        let gamma = model.evaluate(temperature_celsius);
        let initial_pressure_pa = gamma * absorbed_energy_density_j_m3;
        Ok(Self {
            absorbed_energy_density_j_m3,
            initial_pressure_pa,
            confinement,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
    use kwavers_core::constants::thermodynamic::{
        BODY_TEMPERATURE_C, SPECIFIC_HEAT_WATER, THERMAL_CONDUCTIVITY_WATER,
    };

    /// `from_absorbed_energy` ratio matches `GrueneisenModel` ratio.
    ///
    /// For water model, ratio of reports at 37°C vs 20°C must equal
    /// `GrueneisenModel::water().evaluate(37.0) / GrueneisenModel::water().evaluate(20.0)`.
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    #[test]
    fn test_thermoelastic_report_temperature_sensitivity() -> KwaversResult<()> {
        use kwavers_imaging::photoacoustic::ThermoelasticProperties;
        let thermoelastic = ThermoelasticProperties {
            density_kg_m3: DENSITY_WATER_NOMINAL,
            sound_speed_m_s: SOUND_SPEED_WATER_SIM,
            specific_heat_j_kgk: SPECIFIC_HEAT_WATER,
            thermal_conductivity_w_mk: THERMAL_CONDUCTIVITY_WATER,
        };
        let model = GrueneisenModel::water();
        let energy = 1000.0_f64; // J/m³
        let mu_a = 100.0_f64;
        let tau = 5e-9_f64;

        let report_37 = ThermoelasticReport::from_absorbed_energy(
            energy,
            mu_a,
            tau,
            thermoelastic,
            &model,
            BODY_TEMPERATURE_C,
        )?;
        let report_20 = ThermoelasticReport::from_absorbed_energy(
            energy,
            mu_a,
            tau,
            thermoelastic,
            &model,
            20.0,
        )?;

        let gamma_37 = model.evaluate(BODY_TEMPERATURE_C);
        let gamma_20 = model.evaluate(20.0);
        let expected_ratio = gamma_37 / gamma_20;
        let actual_ratio = report_37.initial_pressure_pa / report_20.initial_pressure_pa;

        assert!(
            (actual_ratio - expected_ratio).abs() < 1e-10,
            "p₀ ratio = {actual_ratio:.10}, expected {expected_ratio:.10}"
        );
        // Canonical water ratio: Γ(37)/Γ(20) = 0.188/0.12 ≈ 1.567
        assert!(
            (expected_ratio - 1.567).abs() < 0.001,
            "Water Γ(37)/Γ(20) = {expected_ratio:.4}, expected ≈ 1.567"
        );
        Ok(())
    }

    /// Stress confinement: τ_laser ≪ τ_stress = 1/(μ_a · c_s).
    ///
    /// **Reference:** Oraevsky et al. (1994); τ_stress = δ_a / c_s = 1/(μ_a·c_s).
    /// # Panics
    /// - Panics if the stress confinement condition `τ_laser < τ_stress` is violated.
    ///
    #[test]
    fn test_stress_confinement_condition() {
        let tau_laser = 5e-9_f64; // 5 ns laser pulse
        let mu_a = 100.0_f64; // m⁻¹
        let c_s = SOUND_SPEED_WATER_SIM; // m/s
        let tau_stress = 1.0 / (mu_a * c_s); // = 6.67 µs
        assert!(
            tau_laser < tau_stress,
            "Stress confinement violated: τ_laser = {tau_laser:.2e} s, \
             τ_stress = {tau_stress:.2e} s"
        );
    }
}
