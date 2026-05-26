use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use crate::core::constants::numerical::{MHZ_TO_HZ, MPA_TO_PA};
use std::f64::consts::PI;

use super::*;

#[test]
fn standing_wave_nodes_at_zero() {
    let p = standing_wave_1d(1000.0, 1.0, &[0.0], 0.5);
    assert!((p[0]).abs() < 1e-12);
}

#[test]
fn reflection_plus_transmission_identity() {
    let z1 = 1_480_000.0_f64;
    let z2 = 7_800_000.0_f64;
    let r = reflection_pressure_coeff(z1, z2);
    let t = transmission_pressure_coeff(z1, z2);
    assert!((1.0 + r - t).abs() < 1e-10);
}

#[test]
fn shock_distance_positive() {
    let xs = shock_formation_distance(MPA_TO_PA, MHZ_TO_HZ, SOUND_SPEED_WATER_SIM, 1000.0, 3.5);
    assert!(xs > 0.0);
}

#[test]
fn fdtd_cfl_1d() {
    let cfl = fdtd_cfl_limit(1);
    assert!((cfl - 1.0).abs() < 1e-10);
}

#[test]
fn fdtd_cfl_3d() {
    let cfl = fdtd_cfl_limit(3);
    assert!((cfl - 1.0 / 3.0_f64.sqrt()).abs() < 1e-10);
}

#[test]
fn fubini_n1_sigma0_is_one() {
    // limit: B_1(0) = 1 (from the pre-shock Fubini series)
    let b = fubini_harmonic_amplitude(1, 0.0);
    assert!((b - 1.0).abs() < 1e-8, "b={}", b);
}

#[test]
fn pstd_error_is_zero() {
    let err = pstd_phase_error(&[0.1, 0.5, 1.0, PI]);
    assert!(err.iter().all(|&e| e == 0.0));
}

#[test]
fn westervelt_length_consistency() {
    let z = vec![0.0, 0.01, 0.02];
    let w = westervelt_harmonic_evolution(&z, 1e5, MHZ_TO_HZ, SOUND_SPEED_WATER_SIM, 1000.0, 3.5, 1.0, 3);
    assert_eq!(w.len(), 3);
    assert_eq!(w[0].len(), 3);
}

#[test]
fn fdtd_phase_error_small_kh_near_zero() {
    // For k·Δx → 0, the FDTD dispersion error must vanish (long wavelengths are exact).
    // With CFL = 0.5, the pre-fix formula gave ~100% error even at kh → 0.
    let small_kh = vec![1e-6_f64];
    let err = fdtd_phase_error_1d(&small_kh, 0.5);
    assert!(
        err[0].abs() < 1e-6,
        "Expected near-zero FDTD dispersion error at small kh, got {}",
        err[0]
    );
}

#[test]
fn fdtd_phase_error_cfl_unity_is_zero() {
    // At CFL = 1 the 1-D FDTD scheme is non-dispersive: error = 0 for all kh.
    use std::f64::consts::PI;
    let kh_arr: Vec<f64> = (1..10).map(|i| i as f64 * PI / 10.0).collect();
    let err = fdtd_phase_error_1d(&kh_arr, 1.0);
    for &e in &err {
        assert!(
            e.abs() < 1e-10,
            "Expected zero dispersion at CFL=1, got {}",
            e
        );
    }
}

#[test]
fn stokes_kirchhoff_dc_is_zero() {
    // α_SK(0) = 0: zero absorption at zero frequency.
    let alpha = stokes_kirchhoff_absorption_np_m(&[0.0], 4.33e-6, 1500.0);
    assert_eq!(alpha[0], 0.0);
}

#[test]
fn stokes_kirchhoff_quadratic_scaling() {
    // α_SK ∝ f²: doubling frequency must quadruple absorption.
    let delta = 4.33e-6_f64; // m²/s, water 20°C
    let c0 = 1500.0_f64;
    let alpha = stokes_kirchhoff_absorption_np_m(&[1e6, 2e6], delta, c0);
    let ratio = alpha[1] / alpha[0];
    assert!(
        (ratio - 4.0).abs() < 1e-10,
        "Expected quadratic scaling (ratio=4), got {ratio}"
    );
}

#[test]
fn stokes_kirchhoff_formula_match() {
    // Direct formula check at f = 1 MHz, water 20°C.
    // α = δ·(2πf)²/(2c³)
    let f = 1e6_f64;
    let delta = 4.33e-6_f64;
    let c0 = 1500.0_f64;
    let expected = delta * (2.0 * PI * f).powi(2) / (2.0 * c0 * c0 * c0);
    let alpha = stokes_kirchhoff_absorption_np_m(&[f], delta, c0);
    assert!(
        (alpha[0] - expected).abs() < 1e-20,
        "Formula mismatch: got {}, expected {expected}",
        alpha[0]
    );
}
