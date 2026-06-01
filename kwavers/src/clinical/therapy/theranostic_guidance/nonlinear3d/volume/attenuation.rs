//! CT-derived attenuation maps for nonlinear 3-D theranostic propagation.

use crate::core::constants::acoustic_parameters::NP_TO_DB;
use crate::core::constants::ct_acoustics::HU_BONE_THRESHOLD;
use crate::core::constants::fundamental::ACOUSTIC_ABSORPTION_TISSUE;
use crate::core::constants::tissue_acoustics::{
    ACOUSTIC_ABSORPTION_BRAIN_WHITE, ACOUSTIC_ABSORPTION_SKULL_CORTICAL_MIN,
    ACOUSTIC_ABSORPTION_SKULL_MIN, ACOUSTIC_ABSORPTION_SKULL_RANGE,
};

/// Per-voxel attenuation coefficient at 1 MHz in Np/m. The frequency
/// dependence follows the power law `alpha(f) = alpha(1MHz) * f_MHz^y`, where
/// the per-voxel exponent `y` is returned by `attenuation_power_law_y_from_hu`.
///
/// # Reference values
///
/// Hamilton & Blackstock 1998 Table 4.1 (`alpha_0` in dB / (cm * MHz)):
/// - soft tissue (liver, kidney, muscle, brain): `0.5 - 0.6`
/// - skull bone (cortical): `13 - 20` (Connor & Hynynen 2002)
/// - lung (air-filled): effectively absorbing
///
/// Conversion: `alpha [Np/m] = alpha_0 [dB/(cm*MHz)] * 100 / NP_TO_DB`.
pub(super) fn attenuation_np_per_m_mhz_from_hu(hu: f64, label: i16, body: bool) -> f64 {
    /// Skull absorption upper bound [dB/(cm·MHz)] = ACOUSTIC_ABSORPTION_SKULL_MIN + RANGE = 20.
    const SKULL_ABSORPTION_MAX: f64 =
        ACOUSTIC_ABSORPTION_SKULL_MIN + ACOUSTIC_ABSORPTION_SKULL_RANGE;
    if !body {
        return 0.0;
    }
    if hu >= HU_BONE_THRESHOLD {
        let hu_norm = ((hu - HU_BONE_THRESHOLD) / 1200.0).clamp(0.0, 1.0);
        let alpha_db_cm_mhz = ACOUSTIC_ABSORPTION_SKULL_CORTICAL_MIN
            + (SKULL_ABSORPTION_MAX - ACOUSTIC_ABSORPTION_SKULL_CORTICAL_MIN) * hu_norm;
        alpha_db_cm_mhz * 100.0 / NP_TO_DB
    } else if hu < -700.0 && label == 0 {
        1000.0
    } else if label > 0 {
        ACOUSTIC_ABSORPTION_BRAIN_WHITE * 100.0 / NP_TO_DB
    } else {
        ACOUSTIC_ABSORPTION_TISSUE * 100.0 / NP_TO_DB
    }
}

/// Per-voxel power-law exponent `y` for `alpha(f) = alpha(1MHz) * f_MHz^y`.
///
/// # Reference values
///
/// Treeby & Cox 2010 Table I gives soft-tissue exponents near `1.0 - 1.2`.
/// Connor & Hynynen 2002 reports cortical-skull exponents near `1.9 - 2.0`,
/// matching the Stokes-Kirchhoff classical viscous limit at this abstraction.
pub(super) fn attenuation_power_law_y_from_hu(hu: f64, label: i16, body: bool) -> f64 {
    if !body {
        return 1.0;
    }
    if hu >= HU_BONE_THRESHOLD {
        2.0
    } else if hu < -700.0 && label == 0 {
        1.0
    } else {
        1.05
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NEPER_PER_DB: f64 = 8.685_889_638_065_036;
    const SOFT_TISSUE_ALPHA_NP_PER_M_AT_1MHZ: f64 = 0.5 * 100.0 / NEPER_PER_DB;
    const ORGAN_ALPHA_NP_PER_M_AT_1MHZ: f64 = 0.6 * 100.0 / NEPER_PER_DB;
    const SKULL_BASE_ALPHA_NP_PER_M_AT_1MHZ: f64 = 13.0 * 100.0 / NEPER_PER_DB;
    const SKULL_DENSE_ALPHA_NP_PER_M_AT_1MHZ: f64 = 20.0 * 100.0 / NEPER_PER_DB;

    fn assert_close(actual: f64, expected: f64, abs_tol: f64) {
        assert!(
            (actual - expected).abs() <= abs_tol,
            "expected {expected:.6}, got {actual:.6}, tol={abs_tol:.2e}"
        );
    }

    #[test]
    fn soft_tissue_attenuation_matches_hamilton_blackstock_1998_table_4_1_median() {
        let alpha = attenuation_np_per_m_mhz_from_hu(40.0, 0, true);
        assert_close(alpha, SOFT_TISSUE_ALPHA_NP_PER_M_AT_1MHZ, 1.0e-6);
    }

    #[test]
    fn segmented_organ_attenuation_matches_hamilton_blackstock_1998_organ_median() {
        let alpha = attenuation_np_per_m_mhz_from_hu(60.0, 1, true);
        assert_close(alpha, ORGAN_ALPHA_NP_PER_M_AT_1MHZ, 1.0e-6);
        let alpha_target = attenuation_np_per_m_mhz_from_hu(75.0, 2, true);
        assert_close(alpha_target, ORGAN_ALPHA_NP_PER_M_AT_1MHZ, 1.0e-6);
    }

    #[test]
    fn skull_bone_attenuation_at_lower_hu_bound_matches_connor_hynynen_2002() {
        let alpha = attenuation_np_per_m_mhz_from_hu(300.0, 0, true);
        assert_close(alpha, SKULL_BASE_ALPHA_NP_PER_M_AT_1MHZ, 1.0e-6);
    }

    #[test]
    fn skull_bone_attenuation_at_dense_hu_bound_interpolates_linearly() {
        let alpha = attenuation_np_per_m_mhz_from_hu(1500.0, 0, true);
        assert_close(alpha, SKULL_DENSE_ALPHA_NP_PER_M_AT_1MHZ, 1.0e-6);
        let midpoint = attenuation_np_per_m_mhz_from_hu(900.0, 0, true);
        let expected_midpoint = 16.5 * 100.0 / NEPER_PER_DB;
        assert_close(midpoint, expected_midpoint, 1.0e-6);
    }

    #[test]
    fn air_pocket_attenuation_blocks_propagation() {
        let alpha = attenuation_np_per_m_mhz_from_hu(-900.0, 0, true);
        assert_close(alpha, 1000.0, 1.0e-9);
    }

    #[test]
    fn outside_body_attenuation_is_zero() {
        let alpha = attenuation_np_per_m_mhz_from_hu(40.0, 0, false);
        assert_close(alpha, 0.0, 1.0e-12);
    }

    #[test]
    fn soft_tissue_power_law_y_matches_treeby_cox_2010_table_i() {
        let y = attenuation_power_law_y_from_hu(40.0, 0, true);
        assert_close(y, 1.05, 1.0e-12);
        let y_organ = attenuation_power_law_y_from_hu(60.0, 1, true);
        assert_close(y_organ, 1.05, 1.0e-12);
    }

    #[test]
    fn skull_power_law_y_matches_connor_hynynen_2002_stokes_kirchhoff() {
        let y_low = attenuation_power_law_y_from_hu(300.0, 0, true);
        let y_high = attenuation_power_law_y_from_hu(1500.0, 0, true);
        assert_close(y_low, 2.0, 1.0e-12);
        assert_close(y_high, 2.0, 1.0e-12);
    }

    #[test]
    fn skull_subharmonic_attenuation_with_y2_is_three_times_less_than_y1() {
        let alpha_1mhz = attenuation_np_per_m_mhz_from_hu(500.0, 0, true);
        let y = attenuation_power_law_y_from_hu(500.0, 0, true);
        let f_s_mhz: f64 = 0.325;
        let alpha_y2 = alpha_1mhz * f_s_mhz.powf(y);
        let alpha_y1 = alpha_1mhz * f_s_mhz;
        let ratio = alpha_y1 / alpha_y2;
        let expected_ratio = 1.0 / f_s_mhz;
        assert_close(ratio, expected_ratio, 1.0e-9);
        assert!(
            ratio > 3.0 && ratio < 3.2,
            "y=1 extrapolation overshoots y=2 by {ratio:.2}x at f_s = 325 kHz",
        );
    }
}
