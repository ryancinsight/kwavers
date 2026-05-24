use super::ConfinementAssessment;
use crate::core::constants::thermodynamic::{
    BODY_TEMPERATURE_C, GRUNEISEN_SOFT_TISSUE, GRUNEISEN_SOFT_TISSUE_TEMP_COEFF,
    GRUNEISEN_WATER_20C, GRUNEISEN_WATER_T_REF_C, GRUNEISEN_WATER_TEMP_COEFF,
};
use crate::core::error::KwaversResult;
use crate::domain::imaging::photoacoustic::ThermoelasticProperties;

/// Temperature-dependent Grüneisen parameter Γ(T) = Γ₀ + c_T · (T − T_ref).
///
/// ## Theorem (Sigrist 1986; Xu & Wang 2006, Eq. 2)
///
/// The Grüneisen parameter characterises the thermoelastic conversion efficiency:
///
/// ```text
/// p₀(r) = Γ(T(r)) · μ_a(r) · Φ(r)
/// ```
///
/// Over the physiological temperature range, Γ varies linearly with temperature:
///
/// ```text
/// Γ(T) = Γ₀ + c_T · (T − T_ref)
/// ```
///
/// Canonical values (Sigrist 1986; Jacques 2013):
/// | Medium        | Γ₀   | c_T [K⁻¹] | T_ref [°C] |
/// |---------------|------|-----------|------------|
/// | Water         | 0.12 | 0.004     | 20         |
/// | Soft tissue   | 0.15 | 0.003     | 37         |
///
/// When `d_gamma_d_t` is `None` the model degenerates to the constant value Γ₀,
/// appropriate for homogeneous phantoms or temperature-insensitive media.
///
/// ## References
/// - Sigrist MW (1986). "Laser generation of acoustic waves in liquids and gases."
///   *J Appl Phys* **60**(7), R83. DOI: 10.1063/1.337089
/// - Xu M, Wang LV (2006). "Photoacoustic imaging in biomedicine."
///   *Rev Sci Instrum* **77**, 041101. DOI: 10.1063/1.2195024
/// - Oraevsky AA et al. (1994). "Laser optoacoustic tomography of layered tissues."
///   *Proc SPIE* 2134A.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrueneisenModel {
    /// Baseline Grüneisen parameter Γ₀ (dimensionless, > 0).
    pub gamma_0: f64,
    /// Temperature coefficient c_T = dΓ/dT [K⁻¹]; `None` = temperature-independent.
    pub d_gamma_d_t: Option<f64>,
    /// Reference temperature T_ref [°C] at which Γ₀ applies.
    pub t_ref_celsius: f64,
}

impl GrueneisenModel {
    /// Constant Grüneisen parameter (temperature-independent).
    #[must_use]
    pub fn constant(gamma_0: f64) -> Self {
        Self {
            gamma_0,
            d_gamma_d_t: None,
            t_ref_celsius: 0.0,
        }
    }

    /// Grüneisen model with linear temperature dependence Γ(T) = Γ₀ + c_T·(T − T_ref).
    #[must_use]
    pub fn with_temperature_coefficient(gamma_0: f64, c_t: f64, t_ref: f64) -> Self {
        Self {
            gamma_0,
            d_gamma_d_t: Some(c_t),
            t_ref_celsius: t_ref,
        }
    }

    /// Water model: Γ₀ = GRUNEISEN_WATER_20C, c_T = GRUNEISEN_WATER_TEMP_COEFF K⁻¹,
    /// T_ref = GRUNEISEN_WATER_T_REF_C °C (Sigrist 1986).
    #[must_use]
    pub fn water() -> Self {
        Self::with_temperature_coefficient(
            GRUNEISEN_WATER_20C,
            GRUNEISEN_WATER_TEMP_COEFF,
            GRUNEISEN_WATER_T_REF_C,
        )
    }

    /// Soft-tissue model: Γ₀ = GRUNEISEN_SOFT_TISSUE, c_T = GRUNEISEN_SOFT_TISSUE_TEMP_COEFF K⁻¹,
    /// T_ref = BODY_TEMPERATURE_C (37 °C, Xu & Wang 2006).
    #[must_use]
    pub fn soft_tissue() -> Self {
        Self::with_temperature_coefficient(
            GRUNEISEN_SOFT_TISSUE,
            GRUNEISEN_SOFT_TISSUE_TEMP_COEFF,
            BODY_TEMPERATURE_C,
        )
    }

    /// Evaluate Γ at the given temperature [°C].
    ///
    /// Returns `Γ₀ + c_T · (t_celsius − T_ref)`, or `Γ₀` if the model is
    /// temperature-independent.
    #[must_use]
    pub fn evaluate(&self, t_celsius: f64) -> f64 {
        match self.d_gamma_d_t {
            None => self.gamma_0,
            Some(c_t) => c_t.mul_add(t_celsius - self.t_ref_celsius, self.gamma_0),
        }
    }
}

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
    use crate::core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
    use crate::core::constants::thermodynamic::{
        KELVIN_OFFSET_C, SPECIFIC_HEAT_WATER, THERMAL_CONDUCTIVITY_WATER,
    };

    /// Γ = 0.12 + 0.004·(37−20) = 0.12 + 0.068 = 0.188 for water at 37 °C.
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    #[test]
    fn test_grueneisen_water_at_37c() {
        let model = GrueneisenModel::water();
        let gamma = model.evaluate(BODY_TEMPERATURE_C);
        assert!(
            (gamma - 0.188).abs() < 1e-12,
            "Water Γ(37°C) = {gamma:.12}, expected 0.188"
        );
    }

    /// Temperature-independent model returns Γ₀ for any T.
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    #[test]
    fn test_grueneisen_constant() {
        let model = GrueneisenModel::constant(0.2);
        assert!((model.evaluate(0.0) - 0.2).abs() < 1e-15);
        assert!((model.evaluate(100.0) - 0.2).abs() < 1e-15);
        assert!((model.evaluate(-KELVIN_OFFSET_C) - 0.2).abs() < 1e-15);
    }

    /// p₀(37°C) / p₀(20°C) = Γ(37°C) / Γ(20°C) for soft-tissue model.
    ///
    /// Soft tissue: Γ(20°C) = 0.15 + 0.003·(20−37) = 0.15 − 0.051 = 0.099
    ///              Γ(37°C) = 0.15 + 0.003·(37−37) = 0.15
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    #[test]
    fn test_initial_pressure_temperature_dependence() {
        let model = GrueneisenModel::soft_tissue();
        let gamma_20 = model.evaluate(20.0); // 0.15 - 0.003*17 = 0.099
        let gamma_37 = model.evaluate(BODY_TEMPERATURE_C); // 0.150

        let expected_gamma_20 = 0.15 + 0.003 * (20.0 - BODY_TEMPERATURE_C);
        let expected_gamma_37 = 0.15_f64;
        assert!((gamma_20 - expected_gamma_20).abs() < 1e-12);
        assert!((gamma_37 - expected_gamma_37).abs() < 1e-12);

        // Ratio p₀(37°C)/p₀(20°C) = Γ(37)/Γ(20) (μ_a and Φ cancel)
        let ratio = gamma_37 / gamma_20;
        let expected_ratio = expected_gamma_37 / expected_gamma_20;
        assert!(
            (ratio - expected_ratio).abs() < 1e-10,
            "ratio = {ratio:.10}, expected {expected_ratio:.10}"
        );
    }

    /// `GrueneisenModel::soft_tissue().evaluate(37.0) == 0.15` — canonical value at T_ref.
    ///
    /// This is the SSOT agreement test: the canonical type must return the
    /// published reference value exactly at its own reference temperature.
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    #[test]
    fn test_grueneisen_types_agree_at_body_temperature() {
        // Soft-tissue model: Γ₀ = 0.15, T_ref = 37 °C → Γ(37) = 0.15
        let gamma = GrueneisenModel::soft_tissue().evaluate(BODY_TEMPERATURE_C);
        assert!(
            (gamma - 0.15).abs() < 1e-12,
            "soft_tissue Γ(37°C) = {gamma:.12}, expected 0.15 (Xu & Wang 2006)"
        );
    }

    /// `from_absorbed_energy` ratio matches `GrueneisenModel` ratio.
    ///
    /// For water model, ratio of reports at 37°C vs 20°C must equal
    /// `GrueneisenModel::water().evaluate(37.0) / GrueneisenModel::water().evaluate(20.0)`.
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    #[test]
    fn test_thermoelastic_report_temperature_sensitivity() -> KwaversResult<()> {
        use crate::domain::imaging::photoacoustic::ThermoelasticProperties;
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
