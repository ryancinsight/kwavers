use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};
use kwavers_imaging::photoacoustic::ThermoelasticProperties;

#[derive(Debug, Clone, Copy)]
pub struct ConfinementAssessment {
    pub optical_penetration_depth_m: f64,
    pub stress_confinement_time_s: f64,
    pub thermal_confinement_time_s: f64,
    pub stress_confined: bool,
    pub thermal_confined: bool,
}

impl ConfinementAssessment {
    /// Evaluate stress and thermal confinement for a finite absorbing region.
    ///
    /// ## Theorem
    ///
    /// For optical absorption coefficient `mu_a`, absorber length scale
    /// `delta = 1 / mu_a`, sound speed `c_s`, and thermal diffusivity
    /// `alpha = k / (rho * c_p)`, the characteristic confinement times are
    ///
    /// ```text
    /// tau_s  = delta / c_s
    /// tau_th = delta^2 / (4 * alpha)
    /// ```
    ///
    /// A pulse is stress-confined when `tau_p <= tau_s` and thermally confined
    /// when `tau_p <= tau_th`. The formulas require strictly positive finite
    /// material parameters; invalid domains are rejected rather than regularized
    /// with denominator floors.
    ///
    /// ## References
    /// - Xu M, Wang LV. Photoacoustic imaging in biomedicine. Rev Sci Instrum.
    ///   2006;77:041101. DOI: 10.1063/1.2195024.
    /// - Wang LV, Yao J. A practical guide to photoacoustic tomography in the
    ///   life sciences. Nat Methods. 2016;13:627-638. DOI: 10.1038/nmeth.3925.
    ///
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if any required physical parameter
    ///   is nonfinite or nonpositive.
    pub fn evaluate(
        mu_a_m_inv: f64,
        pulse_duration_s: f64,
        thermoelastic: ThermoelasticProperties,
    ) -> KwaversResult<Self> {
        validate_positive_finite("mu_a_m_inv", mu_a_m_inv)?;
        validate_positive_finite("pulse_duration_s", pulse_duration_s)?;
        validate_positive_finite("density_kg_m3", thermoelastic.density_kg_m3)?;
        validate_positive_finite("sound_speed_m_s", thermoelastic.sound_speed_m_s)?;
        validate_positive_finite("specific_heat_j_kgk", thermoelastic.specific_heat_j_kgk)?;
        validate_positive_finite(
            "thermal_conductivity_w_mk",
            thermoelastic.thermal_conductivity_w_mk,
        )?;

        let optical_penetration_depth_m = 1.0 / mu_a_m_inv;
        let stress_confinement_time_s = optical_penetration_depth_m / thermoelastic.sound_speed_m_s;
        let thermal_diffusivity = thermoelastic.thermal_diffusivity_m2_s();
        validate_positive_finite("thermal_diffusivity_m2_s", thermal_diffusivity)?;
        let thermal_confinement_time_s =
            optical_penetration_depth_m.powi(2) / (4.0 * thermal_diffusivity);

        Ok(Self {
            optical_penetration_depth_m,
            stress_confinement_time_s,
            thermal_confinement_time_s,
            stress_confined: pulse_duration_s <= stress_confinement_time_s,
            thermal_confined: pulse_duration_s <= thermal_confinement_time_s,
        })
    }
}

fn validate_positive_finite(parameter: &str, value: f64) -> KwaversResult<()> {
    if value.is_finite() && value > 0.0 {
        return Ok(());
    }
    Err(KwaversError::Validation(ValidationError::InvalidValue {
        parameter: parameter.to_owned(),
        value,
        reason: "must be finite and strictly positive".to_owned(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
    use kwavers_core::constants::thermodynamic::{
        SPECIFIC_HEAT_WATER_37C, THERMAL_CONDUCTIVITY_WATER_37C,
    };

    fn water() -> ThermoelasticProperties {
        ThermoelasticProperties {
            density_kg_m3: DENSITY_WATER_NOMINAL,
            sound_speed_m_s: SOUND_SPEED_WATER_SIM,
            specific_heat_j_kgk: SPECIFIC_HEAT_WATER_37C,
            thermal_conductivity_w_mk: THERMAL_CONDUCTIVITY_WATER_37C,
        }
    }

    #[test]
    fn evaluates_stress_and_thermal_confinement_without_regularization() -> KwaversResult<()> {
        let assessment = ConfinementAssessment::evaluate(100.0, 5e-9, water())?;
        let expected_delta = 0.01_f64;
        let expected_alpha =
            THERMAL_CONDUCTIVITY_WATER_37C / (DENSITY_WATER_NOMINAL * SPECIFIC_HEAT_WATER_37C);
        let expected_tau_s = expected_delta / SOUND_SPEED_WATER_SIM;
        let expected_tau_th = expected_delta.powi(2) / (4.0 * expected_alpha);

        assert!((assessment.optical_penetration_depth_m - expected_delta).abs() < 1e-15);
        assert!((assessment.stress_confinement_time_s - expected_tau_s).abs() < 1e-15);
        assert!((assessment.thermal_confinement_time_s - expected_tau_th).abs() < 1e-12);
        assert!(assessment.stress_confined);
        assert!(assessment.thermal_confined);
        Ok(())
    }

    #[test]
    fn rejects_nonpositive_absorption_instead_of_infinite_depth() {
        let err = ConfinementAssessment::evaluate(0.0, 5e-9, water()).unwrap_err();
        assert!(matches!(
            err,
            KwaversError::Validation(ValidationError::InvalidValue { parameter, .. })
                if parameter == "mu_a_m_inv"
        ));
    }

    #[test]
    fn rejects_invalid_thermal_transport_instead_of_denominator_floor() {
        let thermoelastic = ThermoelasticProperties {
            thermal_conductivity_w_mk: 0.0,
            ..water()
        };
        let err = ConfinementAssessment::evaluate(100.0, 5e-9, thermoelastic).unwrap_err();
        assert!(matches!(
            err,
            KwaversError::Validation(ValidationError::InvalidValue { parameter, .. })
                if parameter == "thermal_conductivity_w_mk"
        ));
    }
}
