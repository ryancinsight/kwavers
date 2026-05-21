//! Vapor thermal and transport properties

use std::f64::consts::PI;

use super::vapor_pressure::ThermodynamicsCalculator;
use crate::core::constants::fundamental::GAS_CONSTANT as R_GAS;
use crate::core::constants::{M_WATER, P_CRITICAL_WATER, T_CRITICAL_WATER};

impl ThermodynamicsCalculator {
    /// Calculate enthalpy of vaporization at given temperature
    ///
    /// Uses Watson correlation for temperature dependence
    #[must_use]
    pub fn enthalpy_vaporization(&self, temperature: f64) -> f64 {
        // Reference values at normal boiling point
        const H_VAP_NBP: f64 = 40660.0; // J/mol at 100°C
        const T_NBP: f64 = 373.15; // Normal boiling point

        if temperature >= T_CRITICAL_WATER {
            return 0.0;
        }

        // Watson correlation
        let tr = temperature / T_CRITICAL_WATER;
        let tr_nbp = T_NBP / T_CRITICAL_WATER;

        H_VAP_NBP * ((1.0 - tr) / (1.0 - tr_nbp)).powf(0.38)
    }

    /// Calculate specific heat capacity of water vapor
    #[must_use]
    pub fn heat_capacity_vapor(&self, temperature: f64) -> f64 {
        // Shomate equation coefficients for water vapor
        // Valid 500-1700 K (extended range)
        const A: f64 = 30.092;
        const B: f64 = 6.832514;
        const C: f64 = 6.793435;
        const D: f64 = -2.534480;
        const E: f64 = 0.082139;

        let t = temperature / 1000.0;
        (D * t * t).mul_add(t, (C * t).mul_add(t, B.mul_add(t, A))) + E / (t * t)
    }

    /// Calculate thermal conductivity of water vapor
    #[must_use]
    pub fn thermal_conductivity_vapor(&self, temperature: f64, pressure: f64) -> f64 {
        // Water vapor correlation calculation
        // From IAPWS formulation
        let t_reduced = temperature / T_CRITICAL_WATER;
        let p_reduced = pressure / P_CRITICAL_WATER;

        // Base conductivity at low pressure
        let k0 = 0.0181 * (temperature / 300.0).powf(0.76);

        // Pressure correction
        let k_corr = 1.0 + 0.003 * p_reduced / t_reduced;

        k0 * k_corr
    }

    /// Calculate dynamic viscosity of water vapor
    #[must_use]
    pub fn viscosity_vapor(&self, temperature: f64) -> f64 {
        // Sutherland's formula for water vapor
        const MU_REF: f64 = 1.12e-5; // Pa·s at 373 K
        const T_REF: f64 = 373.0; // K
        const S: f64 = 961.0; // Sutherland constant for water vapor

        MU_REF * (temperature / T_REF).powf(1.5) * (T_REF + S) / (temperature + S)
    }

    /// Calculate mass transfer coefficient for evaporation/condensation
    #[must_use]
    pub fn mass_transfer_coefficient(&self, temperature: f64, accommodation_coeff: f64) -> f64 {
        // Hertz-Knudsen equation coefficient
        let molecular_speed = (8.0 * R_GAS * temperature / (PI * M_WATER)).sqrt();
        accommodation_coeff * molecular_speed / 4.0
    }

    /// Calculate equilibrium vapor concentration
    #[must_use]
    pub fn vapor_concentration(&self, temperature: f64, pressure: f64) -> f64 {
        let p_sat = self.vapor_pressure(temperature);
        let mole_fraction = p_sat / pressure;

        // Convert to mass concentration (kg/m³)
        mole_fraction * pressure * M_WATER / (R_GAS * temperature)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::constants::fundamental::ATMOSPHERIC_PRESSURE;
    use crate::core::constants::{T_BOILING_WATER, T_CRITICAL_WATER};

    fn default_calc() -> ThermodynamicsCalculator {
        ThermodynamicsCalculator::default()
    }

    /// `enthalpy_vaporization` at the normal boiling point = H_VAP_NBP = 40660 J/mol.
    ///
    /// Watson correlation: H_vap(T) / H_vap(T_nbp) = ((1−Tr)/(1−Tr_nbp))^0.38.
    /// At T = T_nbp: the ratio = 1.0 → H_vap = H_VAP_NBP exactly.
    #[test]
    fn enthalpy_vaporization_at_nbp_equals_h_vap_nbp() {
        const H_VAP_NBP: f64 = 40660.0;
        let h = default_calc().enthalpy_vaporization(T_BOILING_WATER);
        assert_eq!(h, H_VAP_NBP, "H_vap at T_nbp must equal 40660 J/mol");
    }

    /// `enthalpy_vaporization` at critical temperature = 0.0.
    ///
    /// At the critical point there is no liquid-vapor distinction → L_v = 0.
    #[test]
    fn enthalpy_vaporization_zero_at_critical_temperature() {
        let h = default_calc().enthalpy_vaporization(T_CRITICAL_WATER);
        assert_eq!(h, 0.0, "H_vap must be 0 at T_critical");
    }

    /// `enthalpy_vaporization` decreases monotonically toward T_critical.
    #[test]
    fn enthalpy_vaporization_decreasing_toward_critical() {
        let calc = default_calc();
        let temps = [373.15_f64, 450.0, 530.0, 600.0, 640.0];
        for w in temps.windows(2) {
            let h_lo = calc.enthalpy_vaporization(w[0]);
            let h_hi = calc.enthalpy_vaporization(w[1]);
            assert!(h_lo > h_hi, "H_vap must decrease as T increases toward T_c");
        }
    }

    /// `mass_transfer_coefficient` is positive for physical inputs.
    ///
    /// Hertz-Knudsen: α·√(8RT/(π·M)) / 4 > 0 for T>0, α>0.
    #[test]
    fn mass_transfer_coefficient_positive_for_physical_inputs() {
        let h = default_calc().mass_transfer_coefficient(T_BOILING_WATER, 0.04);
        assert!(
            h > 0.0 && h.is_finite(),
            "mass_transfer_coefficient must be positive finite (got {h:.3e})"
        );
    }

    /// `thermal_conductivity_vapor` is positive and increases with temperature.
    ///
    /// k₀ ∝ T^0.76: conductivity increases with temperature for dilute gases.
    #[test]
    fn thermal_conductivity_vapor_positive_and_increases_with_temperature() {
        let calc = default_calc();
        let p = ATMOSPHERIC_PRESSURE;
        let k_low = calc.thermal_conductivity_vapor(400.0, p);
        let k_high = calc.thermal_conductivity_vapor(600.0, p);
        assert!(k_low > 0.0, "k_vapor must be positive");
        assert!(k_high > k_low, "k_vapor must increase with temperature");
    }

    /// `viscosity_vapor` at T_ref (373 K) = MU_REF (1.12e-5 Pa·s).
    ///
    /// Sutherland: μ(T_ref) = MU_REF × 1.0^1.5 × (T_ref+S)/(T_ref+S) = MU_REF.
    #[test]
    fn viscosity_vapor_at_reference_temperature_equals_mu_ref() {
        const MU_REF: f64 = 1.12e-5;
        const T_REF: f64 = 373.0;
        let mu = default_calc().viscosity_vapor(T_REF);
        assert_eq!(mu, MU_REF, "viscosity at T_ref must equal MU_REF");
    }
}
