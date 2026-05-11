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
