use super::*;
use crate::acoustics::bubble_dynamics::acoustic_pressure::PressureAmplitude;
use kwavers_core::constants::fundamental::{
    ATMOSPHERIC_PRESSURE, DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM,
};
use kwavers_core::constants::numerical::{MHZ_TO_HZ, MPA_TO_PA, TWO_PI};

#[test]
fn minnaert_water_air_bubble() {
    let f = minnaert_resonance_hz(10e-6, 1.4, ATMOSPHERIC_PRESSURE, DENSITY_WATER_NOMINAL);
    assert!((f - 327_000.0).abs() / 327_000.0 < 0.05, "f={}", f);
}

#[test]
fn minnaert_radius_inverse_roundtrips_forward_frequency() {
    let expected_radius_m = 5.0e-6;
    let frequency_hz = minnaert_resonance_hz(
        expected_radius_m,
        1.4,
        ATMOSPHERIC_PRESSURE,
        DENSITY_WATER_NOMINAL,
    );
    let radius_m = minnaert_radius_for_frequency_m(
        frequency_hz,
        1.4,
        ATMOSPHERIC_PRESSURE,
        DENSITY_WATER_NOMINAL,
    );

    // Same closed form solved in opposite directions; 1e-12 relative error is
    // >1000 ulp at the 5 um scale and catches algebraic drift without relying
    // on bitwise equality through sqrt/division ordering.
    assert!(
        (radius_m - expected_radius_m).abs() <= expected_radius_m * 1.0e-12,
        "radius_m={radius_m:.16e} expected={expected_radius_m:.16e}"
    );
}

#[test]
fn minnaert_surface_tension_correction_reduces_to_uncorrected_at_zero_sigma() {
    // σ = 0 ⇒ the corrected form is exactly the large-bubble Minnaert frequency.
    let r0 = 10e-6;
    let uncorrected = minnaert_resonance_hz(r0, 1.4, ATMOSPHERIC_PRESSURE, DENSITY_WATER_NOMINAL);
    let corrected =
        minnaert_resonance_corrected_hz(r0, 1.4, ATMOSPHERIC_PRESSURE, DENSITY_WATER_NOMINAL, 0.0);
    assert!((corrected - uncorrected).abs() <= 1e-9 * uncorrected);
}

#[test]
fn minnaert_surface_tension_correction_matches_closed_form_and_scales_with_radius() {
    const SIGMA: f64 = 0.0725; // water [N/m]
    let (gamma, rho) = (1.4, DENSITY_WATER_NOMINAL);

    // Closed-form check: f₀² = [3γP₀ + (3γ−1)·2σ/R₀] / (ρ·(2πR₀)²).
    let r0 = 1e-6;
    let f = minnaert_resonance_corrected_hz(r0, gamma, ATMOSPHERIC_PRESSURE, rho, SIGMA);
    let laplace = 2.0 * SIGMA / r0;
    let stiffness = 3.0 * gamma * ATMOSPHERIC_PRESSURE + (3.0 * gamma - 1.0) * laplace;
    let expected = (stiffness / rho).sqrt() / (TWO_PI * r0);
    assert!(
        (f - expected).abs() <= 1e-9 * expected,
        "f={f} expected={expected}"
    );

    // Large bubble (R₀ = 1 mm): 2σ/R₀ ≪ P₀ ⇒ correction is negligible (<0.1%).
    // (At 100 µm it is already a non-negligible ~0.5%, consistent with the 1/R₀
    // scaling — surface tension matters increasingly as the bubble shrinks.)
    let r_big = 1e-3;
    let big_corr = minnaert_resonance_corrected_hz(r_big, gamma, ATMOSPHERIC_PRESSURE, rho, SIGMA);
    let big_unc = minnaert_resonance_hz(r_big, gamma, ATMOSPHERIC_PRESSURE, rho);
    assert!(
        (big_corr - big_unc).abs() / big_unc < 1e-3,
        "large-bubble correction must be tiny"
    );

    // Small bubble (R₀ = 1 µm): surface tension raises f₀ by >10% (chapter §5).
    let small_corr = minnaert_resonance_corrected_hz(r0, gamma, ATMOSPHERIC_PRESSURE, rho, SIGMA);
    let small_unc = minnaert_resonance_hz(r0, gamma, ATMOSPHERIC_PRESSURE, rho);
    assert!(
        (small_corr - small_unc) / small_unc > 0.10,
        "sub-micron surface-tension correction must exceed 10%: {small_corr} vs {small_unc}"
    );
    // Invalid σ ⇒ 0.
    assert_eq!(
        minnaert_resonance_corrected_hz(r0, gamma, ATMOSPHERIC_PRESSURE, rho, f64::NAN),
        0.0
    );
}

#[test]
fn closed_form_cavitation_estimators_reject_invalid_domains() {
    assert_eq!(
        minnaert_resonance_hz(0.0, 1.4, ATMOSPHERIC_PRESSURE, DENSITY_WATER_NOMINAL),
        0.0
    );
    assert_eq!(
        minnaert_resonance_hz(10e-6, -1.0, ATMOSPHERIC_PRESSURE, DENSITY_WATER_NOMINAL),
        0.0
    );
    assert_eq!(
        minnaert_radius_for_frequency_m(0.0, 1.4, ATMOSPHERIC_PRESSURE, DENSITY_WATER_NOMINAL),
        0.0
    );
    assert_eq!(
        minnaert_radius_for_frequency_m(1.0e6, 1.4, f64::NAN, DENSITY_WATER_NOMINAL),
        0.0
    );
    // `blake_threshold_pa` is deliberately NOT in this 0.0-sentinel
    // blanket: as a threshold pressure, its non-physical inputs
    // (including NaN) return INFINITY — see
    // `blake_threshold_pa_infinite_for_nonphysical_inputs`.
    assert_eq!(
        rayleigh_collapse_time_s(100e-6, -ATMOSPHERIC_PRESSURE, DENSITY_WATER_NOMINAL),
        0.0
    );
    assert_eq!(
        histotripsy_lesion_radius_m(-1.0, 5e-6, ATMOSPHERIC_PRESSURE, 2000.0),
        0.0
    );
}

#[test]
fn rayleigh_collapse_positive() {
    let tc = rayleigh_collapse_time_s(100e-6, ATMOSPHERIC_PRESSURE, DENSITY_WATER_NOMINAL);
    assert!(tc > 0.0 && tc < 1e-4);
}

#[test]
fn rp_rk4_initial_condition() {
    let t: Vec<f64> = (0..10).map(|i| i as f64 * 1e-9).collect();
    let (r, _) = rayleigh_plesset_rk4(
        10e-6,
        0.0,
        PressureAmplitude::from_pascals(0.0),
        MPA_TO_PA,
        &t,
        ATMOSPHERIC_PRESSURE,
        DENSITY_WATER_NOMINAL,
        0.0725,
        0.001,
        1.4,
        2_330.0,
    );
    assert!((r[0] - 10e-6).abs() < 1e-15);
    assert!((r[9] - 10e-6).abs() / 10e-6 < 0.01);
}

#[test]
fn km_rk4_length_matches() {
    let t: Vec<f64> = (0..5).map(|i| i as f64 * 1e-9).collect();
    let (r, v) = keller_miksis_rk4(
        10e-6,
        0.0,
        PressureAmplitude::from_pascals(0.0),
        MPA_TO_PA,
        &t,
        ATMOSPHERIC_PRESSURE,
        DENSITY_WATER_NOMINAL,
        0.0725,
        0.001,
        1.4,
        2_330.0,
        SOUND_SPEED_WATER_SIM,
    );
    assert_eq!(r.len(), 5);
    assert_eq!(v.len(), 5);
}

#[test]
fn mechanical_index_known_value() {
    let mi = mechanical_index(-MPA_TO_PA, MHZ_TO_HZ);
    assert!((mi - 1.0).abs() < 1e-9, "mi={}", mi);
}

#[test]
fn mechanical_index_scales_inversely_with_sqrt_freq() {
    let mi_1 = mechanical_index(MPA_TO_PA, MHZ_TO_HZ);
    let mi_4 = mechanical_index(MPA_TO_PA, 4.0 * MHZ_TO_HZ);
    assert!((mi_1 / mi_4 - 2.0).abs() < 1e-9, "ratio={}", mi_1 / mi_4);
}

#[test]
fn mechanical_index_rejects_invalid_domain() {
    assert_eq!(mechanical_index(MPA_TO_PA, 0.0), 0.0);
    assert_eq!(mechanical_index(MPA_TO_PA, -MHZ_TO_HZ), 0.0);
    assert_eq!(mechanical_index(f64::NAN, MHZ_TO_HZ), 0.0);
}

#[test]
fn icd_zero_driving_gives_zero() {
    let t: Vec<f64> = (0..100).map(|i| i as f64 * 1e-9).collect();
    let (r, rdot) = rayleigh_plesset_rk4(
        10e-6,
        0.0,
        PressureAmplitude::from_pascals(0.0),
        MPA_TO_PA,
        &t,
        ATMOSPHERIC_PRESSURE,
        DENSITY_WATER_NOMINAL,
        0.0725,
        0.001,
        1.4,
        2_330.0,
    );
    let icd = inertial_cavitation_dose(&r, &rdot, 10e-6);
    assert_eq!(icd, 0.0, "icd={}", icd);
}

#[test]
fn icd_strong_driving_nonzero() {
    let f0 = 500e3_f64;
    let n_pts = 2000usize;
    let dt = 1.0 / (20.0 * f0);
    let t: Vec<f64> = (0..n_pts).map(|i| i as f64 * dt).collect();
    let r0 = 5e-6;
    let (r, rdot) = rayleigh_plesset_rk4(
        r0,
        0.0,
        PressureAmplitude::from_pascals(5e6),
        f0,
        &t,
        ATMOSPHERIC_PRESSURE,
        DENSITY_WATER_NOMINAL,
        0.0725,
        0.001,
        1.4,
        2_330.0,
    );
    let icd = inertial_cavitation_dose(&r, &rdot, r0);
    assert!(
        icd > 0.0,
        "expected ICD > 0 for strong driving, got {}",
        icd
    );
}

#[test]
fn lesion_radius_scales_with_icd_cube_root() {
    let r0 = 5e-6;
    let p0 = ATMOSPHERIC_PRESSURE;
    let sigma_y = 2000.0;
    let r1 = histotripsy_lesion_radius_m(1.0, r0, p0, sigma_y);
    let r8 = histotripsy_lesion_radius_m(8.0, r0, p0, sigma_y);
    assert!((r8 / r1 - 2.0).abs() < 1e-9, "r8/r1={}", r8 / r1);
}

#[test]
fn lesion_radius_dimensional_consistency() {
    let r0 = 5e-6;
    let p0 = ATMOSPHERIC_PRESSURE;
    let sigma_y = ATMOSPHERIC_PRESSURE;
    let r_l = histotripsy_lesion_radius_m(1.0, r0, p0, sigma_y);
    assert!((r_l - r0).abs() < 1e-18, "r_l={}, r0={}", r_l, r0);
}

#[test]
fn period_doubling_ratio_no_subharmonic_is_small() {
    let n = 512usize;
    let f0 = MHZ_TO_HZ;
    let fs = f0 * n as f64 / 8.0;
    let dt = 1.0 / fs;
    let r: Vec<f64> = (0..n)
        .map(|i| 1e-7 * (TWO_PI * f0 * i as f64 * dt).sin())
        .collect();
    let (f_arr, p_arr) = bubble_power_spectrum(&r, dt, n);
    let pd = period_doubling_ratio(&f_arr, &p_arr, f0);
    assert!(pd < 0.1, "expected near-zero PD ratio, got {}", pd);
}

#[test]
fn period_doubling_ratio_dominant_subharmonic_exceeds_one() {
    let n = 512usize;
    let f0 = MHZ_TO_HZ;
    let fs = f0 * n as f64 / 8.0;
    let dt = 1.0 / fs;
    let r: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64 * dt;
            1e-7 * (TWO_PI * f0 * t).sin() + 3e-7 * (TWO_PI * f0 * 0.5 * t).sin()
        })
        .collect();
    let (f_arr, p_arr) = bubble_power_spectrum(&r, dt, n);
    let pd = period_doubling_ratio(&f_arr, &p_arr, f0);
    assert!(
        pd > 1.0,
        "expected PD ratio > 1 for dominant subharmonic, got {}",
        pd
    );
}

#[test]
fn thermal_threshold_correction_lowers_threshold_above_baseline_only() {
    let slope = 0.3e6; // 0.3 MPa/°C (book §32.5.2)
    let t_ref = 20.0;
    // At or below the 20 °C baseline: no correction.
    assert_eq!(
        intrinsic_threshold_thermal_correction_pa(20.0, slope, t_ref),
        0.0
    );
    assert_eq!(
        intrinsic_threshold_thermal_correction_pa(10.0, slope, t_ref),
        0.0
    );
    // 37 °C body temperature: −0.3 MPa/°C × 17 °C = −5.1 MPa.
    assert!((intrinsic_threshold_thermal_correction_pa(37.0, slope, t_ref) + 5.1e6).abs() < 1.0);
    // Composes additively with the frequency model: p_T(0.5 MHz, 37 °C) = p_T(0.5 MHz) − 5.1 MPa.
    let pt_freq = frequency_dependent_intrinsic_threshold_pa(&[0.5e6], 28.2e6, 1.4e6)[0];
    let pt_ft = pt_freq + intrinsic_threshold_thermal_correction_pa(37.0, slope, t_ref);
    assert!((pt_ft - (pt_freq - 5.1e6)).abs() < 1.0);
    // Strictly more negative (lower threshold) as temperature rises.
    assert!(
        intrinsic_threshold_thermal_correction_pa(50.0, slope, t_ref)
            < intrinsic_threshold_thermal_correction_pa(40.0, slope, t_ref)
    );
}

#[test]
fn bubble_spectrum_length() {
    let r: Vec<f64> = (0..64)
        .map(|i| 10e-6 + 1e-7 * (i as f64 * 0.1).sin())
        .collect();
    let (f, p) = bubble_power_spectrum(&r, 1e-9, 64);
    assert_eq!(f.len(), 33);
    assert_eq!(p.len(), 33);
    assert_eq!(f[0], 0.0);
    assert!(p.iter().all(|&v| v >= 0.0));
}

/// Blake threshold guard: a non-positive nucleus radius or a negative
/// surface tension has no finite threshold — INFINITY, not this module's
/// 0.0 invalid-input sentinel (0.0 Pa would claim cavitation at zero
/// acoustic amplitude). σ = 0 stays on the formula path, where the closed
/// form reduces to √2·P₀. NaN inputs take the same INFINITY path
/// (physics review 2026-11-27): their comparisons are all false, so the
/// pre-review guard waved them through to the 0.0 finiteness fallback.
#[test]
fn blake_threshold_pa_infinite_for_nonphysical_inputs() {
    assert!(blake_threshold_pa(0.0, ATMOSPHERIC_PRESSURE, 0.0728).is_infinite());
    assert!(blake_threshold_pa(-5e-6, ATMOSPHERIC_PRESSURE, 0.0728).is_infinite());
    assert!(blake_threshold_pa(5e-6, ATMOSPHERIC_PRESSURE, -0.0728).is_infinite());
    assert!(blake_threshold_pa(f64::NAN, ATMOSPHERIC_PRESSURE, 0.0728).is_infinite());
    assert!(blake_threshold_pa(5e-6, ATMOSPHERIC_PRESSURE, f64::NAN).is_infinite());
    assert!(blake_threshold_pa(5e-6, f64::NAN, 0.0728).is_infinite());
    let sigma_zero = blake_threshold_pa(5e-6, ATMOSPHERIC_PRESSURE, 0.0);
    assert!(
        (sigma_zero - ATMOSPHERIC_PRESSURE * std::f64::consts::SQRT_2).abs() < 1.0,
        "σ = 0 must stay on the formula path: got {sigma_zero:.1}"
    );
}
