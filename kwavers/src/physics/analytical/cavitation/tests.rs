use std::f64::consts::PI;

use crate::core::constants::fundamental::{ATMOSPHERIC_PRESSURE, SOUND_SPEED_WATER_SIM};
use super::*;

#[test]
fn minnaert_water_air_bubble() {
    let f = minnaert_resonance_hz(10e-6, 1.4, ATMOSPHERIC_PRESSURE, 998.0);
    assert!((f - 327_000.0).abs() / 327_000.0 < 0.05, "f={}", f);
}

#[test]
fn closed_form_cavitation_estimators_reject_invalid_domains() {
    assert_eq!(minnaert_resonance_hz(0.0, 1.4, ATMOSPHERIC_PRESSURE, 998.0), 0.0);
    assert_eq!(minnaert_resonance_hz(10e-6, -1.0, ATMOSPHERIC_PRESSURE, 998.0), 0.0);
    assert_eq!(blake_threshold_pa(10e-6, ATMOSPHERIC_PRESSURE, f64::NAN), 0.0);
    assert_eq!(rayleigh_collapse_time_s(100e-6, -ATMOSPHERIC_PRESSURE, 998.0), 0.0);
    assert_eq!(
        histotripsy_lesion_radius_m(-1.0, 5e-6, ATMOSPHERIC_PRESSURE, 2000.0),
        0.0
    );
}

#[test]
fn rayleigh_collapse_positive() {
    let tc = rayleigh_collapse_time_s(100e-6, ATMOSPHERIC_PRESSURE, 998.0);
    assert!(tc > 0.0 && tc < 1e-4);
}

#[test]
fn rp_rk4_initial_condition() {
    let t: Vec<f64> = (0..10).map(|i| i as f64 * 1e-9).collect();
    let (r, _) = rayleigh_plesset_rk4(
        10e-6, 0.0, 0.0, 1e6, &t, ATMOSPHERIC_PRESSURE, 998.0, 0.0725, 0.001, 1.4, 2_330.0,
    );
    assert!((r[0] - 10e-6).abs() < 1e-15);
    assert!((r[9] - 10e-6).abs() / 10e-6 < 0.01);
}

#[test]
fn km_rk4_length_matches() {
    let t: Vec<f64> = (0..5).map(|i| i as f64 * 1e-9).collect();
    let (r, v) = keller_miksis_rk4(
        10e-6, 0.0, 0.0, 1e6, &t, ATMOSPHERIC_PRESSURE, 998.0, 0.0725, 0.001, 1.4, 2_330.0, SOUND_SPEED_WATER_SIM,
    );
    assert_eq!(r.len(), 5);
    assert_eq!(v.len(), 5);
}

#[test]
fn mechanical_index_known_value() {
    let mi = mechanical_index(-1e6, 1e6);
    assert!((mi - 1.0).abs() < 1e-9, "mi={}", mi);
}

#[test]
fn mechanical_index_scales_inversely_with_sqrt_freq() {
    let mi_1 = mechanical_index(1e6, 1e6);
    let mi_4 = mechanical_index(1e6, 4e6);
    assert!((mi_1 / mi_4 - 2.0).abs() < 1e-9, "ratio={}", mi_1 / mi_4);
}

#[test]
fn mechanical_index_rejects_invalid_domain() {
    assert_eq!(mechanical_index(1e6, 0.0), 0.0);
    assert_eq!(mechanical_index(1e6, -1e6), 0.0);
    assert_eq!(mechanical_index(f64::NAN, 1e6), 0.0);
}

#[test]
fn icd_zero_driving_gives_zero() {
    let t: Vec<f64> = (0..100).map(|i| i as f64 * 1e-9).collect();
    let (r, rdot) = rayleigh_plesset_rk4(
        10e-6, 0.0, 0.0, 1e6, &t, ATMOSPHERIC_PRESSURE, 998.0, 0.0725, 0.001, 1.4, 2_330.0,
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
        r0, 0.0, 5e6, f0, &t, ATMOSPHERIC_PRESSURE, 998.0, 0.0725, 0.001, 1.4, 2_330.0,
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
    let f0 = 1e6_f64;
    let fs = f0 * n as f64 / 8.0;
    let dt = 1.0 / fs;
    let r: Vec<f64> = (0..n)
        .map(|i| 1e-7 * (2.0 * PI * f0 * i as f64 * dt).sin())
        .collect();
    let (f_arr, p_arr) = bubble_power_spectrum(&r, dt, n);
    let pd = period_doubling_ratio(&f_arr, &p_arr, f0);
    assert!(pd < 0.1, "expected near-zero PD ratio, got {}", pd);
}

#[test]
fn period_doubling_ratio_dominant_subharmonic_exceeds_one() {
    let n = 512usize;
    let f0 = 1e6_f64;
    let fs = f0 * n as f64 / 8.0;
    let dt = 1.0 / fs;
    let r: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64 * dt;
            1e-7 * (2.0 * PI * f0 * t).sin() + 3e-7 * (2.0 * PI * f0 * 0.5 * t).sin()
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
fn bubble_spectrum_length() {
    let r: Vec<f64> = (0..64)
        .map(|i| 10e-6 + 1e-7 * (i as f64 * 0.1).sin())
        .collect();
    let (f, p) = bubble_power_spectrum(&r, 1e-9, 64);
    assert_eq!(f.len(), 33);
    assert_eq!(p.len(), 33);
    assert!(f[0] == 0.0);
    assert!(p.iter().all(|&v| v >= 0.0));
}
