//! Water phase-change property laws used by Keller-Miksis thermodynamics.

const ANTOINE_A: f64 = 8.07131;
const ANTOINE_B: f64 = 1730.63;
const ANTOINE_C: f64 = 233.426;
const MMHG_TO_PA: f64 = 133.322;

/// Saturation vapor pressure of water at `t_celsius` using the Antoine equation.
///
/// Valid for 1-100 C. Returns pressure in Pa.
#[must_use]
pub fn p_sat_water_pa(t_celsius: f64) -> f64 {
    let log_p_mmhg = ANTOINE_A - ANTOINE_B / (ANTOINE_C + t_celsius);
    10f64.powf(log_p_mmhg) * MMHG_TO_PA
}

/// Temperature-dependent latent heat of vaporization for water.
///
/// Linear Watson/NIST fit:
///
/// ```text
/// L_v(T) = 2.501e6 - 2369 T_C  [J/kg]
/// ```
#[must_use]
pub fn latent_heat_water_j_per_kg(t_celsius: f64) -> f64 {
    2369.0f64.mul_add(-t_celsius, 2.501e6)
}

#[cfg(test)]
mod tests {
    use crate::core::constants::fundamental::ATMOSPHERIC_PRESSURE;
    use super::*;

    // Antoine constants for verification: A=8.07131, B=1730.63, C=233.426
    // Unit conversion: 1 mmHg = 133.322 Pa

    /// `p_sat_water_pa` at 100 °C equals 1 atm (101 325 Pa) within 0.5%.
    ///
    /// At 100 °C the Antoine equation gives the standard boiling point pressure.
    /// log₁₀(P_mmHg) = 8.07131 − 1730.63/(233.426+100) = 2.8815... → ~760 mmHg → ~101 kPa.
    #[test]
    fn p_sat_at_100c_equals_one_atmosphere() {
        let p = p_sat_water_pa(100.0);
        // Standard atmospheric pressure within 0.5%
        assert!(
            (p - ATMOSPHERIC_PRESSURE).abs() / ATMOSPHERIC_PRESSURE < 0.005,
            "p_sat(100°C) must be ≈101325 Pa (got {p:.1})"
        );
    }

    /// `p_sat_water_pa` at 20 °C ≈ 2330 Pa (standard vapor pressure of water).
    ///
    /// Reference: NIST Chemistry WebBook, water at 20 °C = 2338.5 Pa.
    #[test]
    fn p_sat_at_20c_matches_standard_value() {
        let p = p_sat_water_pa(20.0);
        // Within 1% of NIST value 2338.5 Pa
        assert!(
            (p - 2338.5).abs() / 2338.5 < 0.01,
            "p_sat(20°C) must be ≈2339 Pa (got {p:.1})"
        );
    }

    /// `p_sat_water_pa` is monotonically increasing with temperature.
    ///
    /// Clausius-Clapeyron: dP/dT > 0 for liquid→vapor transition.
    #[test]
    fn p_sat_monotonically_increasing_with_temperature() {
        let temps = [0.0_f64, 20.0, 40.0, 60.0, 80.0, 100.0];
        for w in temps.windows(2) {
            let p_low = p_sat_water_pa(w[0]);
            let p_high = p_sat_water_pa(w[1]);
            assert!(
                p_high > p_low,
                "p_sat must increase with T: p({})={p_low:.1} < p({})={p_high:.1}",
                w[0],
                w[1]
            );
        }
    }

    /// `latent_heat_water_j_per_kg` at 0 °C = 2.501 MJ/kg (freezing point).
    ///
    /// Reference: NIST / CRC Handbook: L_v(0°C) = 2500.9 kJ/kg.
    #[test]
    fn latent_heat_at_0c_equals_defined_constant() {
        let lv = latent_heat_water_j_per_kg(0.0);
        // By definition of the linear model: 2.501e6 − 2369×0 = 2.501e6 J/kg
        assert_eq!(lv, 2.501e6, "L_v(0°C) must be exactly 2.501e6 J/kg");
    }

    /// `latent_heat_water_j_per_kg` at 100 °C ≈ 2.264 MJ/kg (boiling point).
    ///
    /// Reference: NIST: L_v(100°C) = 2256.9 kJ/kg.
    /// Model: 2.501e6 − 2369×100 = 2264100 J/kg ≈ 2.264 MJ/kg.
    #[test]
    fn latent_heat_at_100c_close_to_2264_kj_per_kg() {
        let lv = latent_heat_water_j_per_kg(100.0);
        assert_eq!(lv, 2.501e6 - 2369.0 * 100.0, "exact model formula");
        // Within 0.4% of NIST reference 2256900 J/kg
        assert!(
            (lv - 2_256_900.0).abs() / 2_256_900.0 < 0.004,
            "L_v(100°C) must be ≈2257 kJ/kg (got {:.1} kJ/kg)",
            lv / 1000.0
        );
    }

    /// `latent_heat_water_j_per_kg` decreases linearly with temperature.
    ///
    /// Slope: −2369 J/(kg·°C). Two-point verification.
    #[test]
    fn latent_heat_decreases_linearly_with_temperature() {
        let lv0 = latent_heat_water_j_per_kg(0.0);
        let lv1 = latent_heat_water_j_per_kg(1.0);
        let slope = lv1 - lv0;
        assert!(
            (slope - (-2369.0)).abs() < 1e-9,
            "slope must be -2369 J/(kg·°C) (got {slope:.3})"
        );
    }
}
