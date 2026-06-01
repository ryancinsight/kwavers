//! Ultrasound dosimetry and safety indices for book chapter ch15.
//!
//! Covers: Mechanical Index (MI), Thermal Index soft tissue (TIS) and bone
//! (TIB), CEM43 cumulative dose, Arrhenius damage integral, and FDA output
//! limits.

use crate::core::constants::fundamental::GAS_CONSTANT;
use crate::core::constants::medical::{
    IEC_TIB_DIVISOR, IEC_TIS_DIVISOR, THERMAL_DOSE_REFERENCE_TEMP_C, THERMAL_DOSE_R_ABOVE_43C,
    THERMAL_DOSE_R_BELOW_43C,
};
use crate::core::constants::numerical::SECONDS_PER_MINUTE;
use crate::core::constants::thermodynamic::KELVIN_OFFSET_C;

/// Mechanical Index (MI).
///
/// ```text
/// MI = |p_r| / (1e6 · √(f_MHz))   [dimensionless]
/// ```
/// where `p_r` is the peak rarefactional pressure in Pa and
/// `f_MHz = f_hz / 1e6`.
///
/// FDA limit: MI ≤ 1.9 (general) or 0.23 (ophthalmic).
///
/// Delegates to the canonical
/// [`crate::physics::acoustics::analysis::calculate_mechanical_index`] so the
/// book-chapter API and the production safety paths share a single contract.
///
/// # Reference
/// FDA, *Marketing Clearance of Diagnostic Ultrasound Systems and
/// Transducers*, Appendix A; IEC 62359 (2017) §7.2.
#[inline]
pub fn mechanical_index(p_neg_pa: f64, f_hz: f64) -> f64 {
    crate::physics::acoustics::analysis::calculate_mechanical_index(p_neg_pa, f_hz)
}

/// Mechanical Index over a pressure field (array variant).
///
/// Applies [`mechanical_index`] element-wise to every sample in `p_field`:
/// ```text
/// MI_i = |p_field[i]| / (1e6 · √(f_MHz))
/// ```
///
/// # Arguments
/// * `p_field` – peak rarefactional pressures [Pa], any shape passed as 1-D
/// * `f_hz` – centre frequency [Hz]
///
/// # Reference
/// FDA, *Marketing Clearance of Diagnostic Ultrasound Systems and
/// Transducers*, Appendix A; IEC 62359 (2017) §7.2.
#[must_use]
#[inline]
pub fn mechanical_index_field(p_field: &[f64], f_hz: f64) -> Vec<f64> {
    p_field.iter().map(|&p| mechanical_index(p, f_hz)).collect()
}

/// Thermal Index for soft tissue (TIS).
///
/// Simplified IEC 62359 formula:
/// ```text
/// TIS = W_stp [mW] / (210 · f_MHz)
/// ```
///
/// # Arguments
/// * `wstp_mw` – spatial-temporal peak power at the focus [mW]
/// * `f_mhz` – centre frequency [MHz]
///
/// # Reference
/// IEC 62359 (2017) §8.3.2.
#[inline]
pub fn thermal_index_soft_tissue(wstp_mw: f64, f_mhz: f64) -> f64 {
    if !(wstp_mw.is_finite() && f_mhz.is_finite() && wstp_mw >= 0.0 && f_mhz > 0.0) {
        return 0.0;
    }
    wstp_mw / (IEC_TIS_DIVISOR * f_mhz)
}

/// Thermal Index for bone (TIB).
///
/// Simplified formula:
/// ```text
/// TIB = W [mW] · f_MHz / 40.0
/// ```
///
/// # Reference
/// IEC 62359 (2017) §8.4.
#[inline]
pub fn thermal_index_bone(w_mw: f64, f_mhz: f64) -> f64 {
    if !(w_mw.is_finite() && f_mhz.is_finite() && w_mw >= 0.0 && f_mhz >= 0.0) {
        return 0.0;
    }
    w_mw * f_mhz / IEC_TIB_DIVISOR
}

/// CEM43 cumulative equivalent minutes at 43 °C.
///
/// Computed at each time step i as the running sum:
/// ```text
/// CEM43[i] = Σ_{j=0}^{i} (dt/60) · R^(43 − T[j])
/// R = 0.5  if T[j] ≥ 43 °C
/// R = 0.25 if T[j] < 43 °C
/// ```
///
/// # Arguments
/// * `t_celsius` – temperature time series [°C]
/// * `dt_s` – time step [s]
///
/// Returns cumulative CEM43 at each time step [min].
///
/// # Reference
/// Sapareto & Dewey (1984), *Int. J. Radiat. Oncol. Biol. Phys.* 10, 787.
#[must_use]
pub fn cem43_cumulative(t_celsius: &[f64], dt_s: f64) -> Vec<f64> {
    let dt_min = dt_s / SECONDS_PER_MINUTE;
    let mut cem = 0.0_f64;
    t_celsius
        .iter()
        .map(|&t| {
            let r: f64 = if t >= THERMAL_DOSE_REFERENCE_TEMP_C {
                THERMAL_DOSE_R_ABOVE_43C
            } else {
                THERMAL_DOSE_R_BELOW_43C
            };
            cem += dt_min * r.powf(THERMAL_DOSE_REFERENCE_TEMP_C - t);
            cem
        })
        .collect()
}

/// Arrhenius thermal damage integral.
///
/// ```text
/// Ω = A · ∫ exp(−Ea / (R_gas · T_K(t))) dt
/// ```
/// computed by the rectangle rule:
/// ```text
/// Ω ≈ A · Σ_i exp(−Ea / (R_gas · (T[i] + 273.15))) · dt
/// ```
/// `Ω ≥ 1` indicates irreversible damage (63% cell kill at Ω = 1).
///
/// # Arguments
/// * `t_celsius` – temperature time series [°C]
/// * `dt_s` – time step [s]
/// * `a_per_s` – frequency factor A [s⁻¹]
/// * `ea_j_mol` – activation energy Ea [J/mol]
///
/// # Reference
/// Henriques & Moritz (1947), *Am. J. Pathol.* 23, 531;
/// Bhowmick et al. (2002) for muscle tissue parameters.
pub fn arrhenius_damage_integral(t_celsius: &[f64], dt_s: f64, a_per_s: f64, ea_j_mol: f64) -> f64 {
    let r_gas = GAS_CONSTANT;
    t_celsius.iter().fold(0.0_f64, |acc, &t| {
        let t_k = t + KELVIN_OFFSET_C;
        acc + a_per_s * (-ea_j_mol / (r_gas * t_k)).exp() * dt_s
    })
}

/// Cumulative Arrhenius thermal-damage integral over a temperature time series.
///
/// Returns the running sum Ω(t_k) at each discrete time step:
/// ```text
/// Ω(t_k) = A · Σ_{i=0}^{k} exp(−Ea / (R_gas · T_K[i])) · dt
/// ```
/// The output has the same length as `t_celsius`; element `k` is the
/// total damage accumulated from t=0 through t=k·dt.
///
/// `Ω ≥ 1` indicates irreversible damage (Henriques criterion).
///
/// # Arguments
/// * `t_celsius` – temperature time series [°C]
/// * `dt_s` – time step [s]
/// * `a_per_s` – frequency factor A [s⁻¹]
/// * `ea_j_mol` – activation energy Ea [J/mol]
///
/// # Reference
/// Henriques & Moritz (1947), *Am. J. Pathol.* 23, 531.
#[must_use]
pub fn arrhenius_cumulative(t_celsius: &[f64], dt_s: f64, a_per_s: f64, ea_j_mol: f64) -> Vec<f64> {
    let r_gas = GAS_CONSTANT;
    let mut acc = 0.0_f64;
    t_celsius
        .iter()
        .map(|&t| {
            let t_k = t + KELVIN_OFFSET_C;
            acc += a_per_s * (-ea_j_mol / (r_gas * t_k)).exp() * dt_s;
            acc
        })
        .collect()
}

/// FDA ISPTA.3 output limit.
///
/// ```text
/// ISPTA.3 ≤ 720 mW/cm²
/// ```
///
/// # Reference
/// FDA, *Marketing Clearance of Diagnostic Ultrasound Systems and
/// Transducers*, Table 3.
#[inline]
pub fn fda_ispta_limit_mw_cm2() -> f64 {
    720.0
}

/// FDA ISPPA.3 output limit.
///
/// ```text
/// ISPPA.3 ≤ 190 W/cm²
/// ```
///
/// # Reference
/// FDA, *Marketing Clearance of Diagnostic Ultrasound Systems and
/// Transducers*, Table 3.
#[inline]
pub fn fda_isppa_limit_w_cm2() -> f64 {
    190.0
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::constants::medical::THERMAL_DOSE_REFERENCE_TEMP_C;
    use crate::core::constants::numerical::{MHZ_TO_HZ, MPA_TO_PA};
    use crate::core::constants::thermodynamic::BODY_TEMPERATURE_C;

    #[test]
    fn mi_dimensional_check() {
        // At 1 MPa negative, 1 MHz: MI = 1.0
        let mi = mechanical_index(-MPA_TO_PA, MHZ_TO_HZ);
        assert!((mi - 1.0).abs() < 1e-10);
    }

    #[test]
    fn mi_rejects_nonpositive_frequency() {
        assert_eq!(mechanical_index(-MPA_TO_PA, 0.0), 0.0);
        assert_eq!(mechanical_index(-MPA_TO_PA, -MHZ_TO_HZ), 0.0);
    }

    #[test]
    fn thermal_indices_are_nonnegative_ratios() {
        assert!((thermal_index_soft_tissue(210.0, 1.0) - 1.0).abs() < 1e-12);
        assert!((thermal_index_bone(40.0, 1.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn thermal_indices_reject_invalid_domains() {
        assert_eq!(thermal_index_soft_tissue(1.0, 0.0), 0.0);
        assert_eq!(thermal_index_soft_tissue(-1.0, 1.0), 0.0);
        assert_eq!(thermal_index_soft_tissue(f64::NAN, 1.0), 0.0);
        assert_eq!(thermal_index_bone(-1.0, 1.0), 0.0);
        assert_eq!(thermal_index_bone(1.0, -1.0), 0.0);
        assert_eq!(thermal_index_bone(1.0, f64::INFINITY), 0.0);
    }

    #[test]
    fn cem43_flat_43c() {
        // At exactly 43°C, R = 0.5, 0.5^0 = 1 → each step contributes dt/60
        let t = vec![THERMAL_DOSE_REFERENCE_TEMP_C; 6];
        let dt = 10.0; // seconds
        let c = cem43_cumulative(&t, dt);
        let expected: Vec<f64> = (1..=6).map(|i| i as f64 * dt / 60.0).collect();
        for (got, exp) in c.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-10, "got={} exp={}", got, exp);
        }
    }

    #[test]
    fn cem43_below_threshold_smaller() {
        // Below 43°C (R=0.25), increments are smaller than at 43°C
        let t_above = vec![THERMAL_DOSE_REFERENCE_TEMP_C; 3];
        let t_below = vec![BODY_TEMPERATURE_C; 3];
        let dt = 1.0;
        let c_above = cem43_cumulative(&t_above, dt);
        let c_below = cem43_cumulative(&t_below, dt);
        assert!(c_below.last().unwrap() < c_above.last().unwrap());
    }

    #[test]
    fn arrhenius_positive() {
        // 70°C for 1 s with muscle parameters (A ≈ 3.1e98, Ea ≈ 6.28e5 J/mol)
        let t = vec![70.0; 100];
        let omega = arrhenius_damage_integral(&t, 0.01, 3.1e98, 6.28e5);
        assert!(omega > 0.0);
    }

    #[test]
    fn fda_limits_positive() {
        assert!(fda_ispta_limit_mw_cm2() > 0.0);
        assert!(fda_isppa_limit_w_cm2() > 0.0);
    }

    #[test]
    fn mechanical_index_field_matches_scalar() {
        // Field variant must produce the same result as the scalar variant per element.
        let pressures = [-MPA_TO_PA, -2.0 * MPA_TO_PA, -0.5 * MPA_TO_PA];
        let f_hz = 3.0 * MHZ_TO_HZ;
        let field = mechanical_index_field(&pressures, f_hz);
        for (&p, &mi_field) in pressures.iter().zip(field.iter()) {
            let mi_scalar = mechanical_index(p, f_hz);
            assert!(
                (mi_field - mi_scalar).abs() < 1e-12,
                "field={mi_field} scalar={mi_scalar}"
            );
        }
    }
}
