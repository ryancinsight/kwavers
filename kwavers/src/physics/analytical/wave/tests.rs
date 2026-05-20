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
    let xs = shock_formation_distance(1e6, 1e6, 1500.0, 1000.0, 3.5);
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
    let w = westervelt_harmonic_evolution(&z, 1e5, 1e6, 1500.0, 1000.0, 3.5, 1.0, 3);
    assert_eq!(w.len(), 3);
    assert_eq!(w[0].len(), 3);
}
