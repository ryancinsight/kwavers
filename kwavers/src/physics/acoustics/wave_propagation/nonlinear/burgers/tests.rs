use super::*;
use crate::physics::acoustics::wave_propagation::NonlinearParameters;
use std::f64::consts::PI;

#[test]
fn bessel_j0_at_zero() {
    assert_eq!(bessel_j(0, 0.0), 1.0);
}

#[test]
fn bessel_jn_at_zero_for_n_gt_0() {
    for n in 1..=5 {
        assert_eq!(bessel_j(n, 0.0), 0.0, "J_{n}(0) must be 0");
    }
}

#[test]
fn bessel_j1_small_argument() {
    let val = bessel_j(1, 0.5);
    assert!(
        (val - 0.24227_f64).abs() < 1e-5,
        "J1(0.5) = {val}, expected about 0.24227"
    );
}

#[test]
fn bessel_j2_unit_argument() {
    let val = bessel_j(2, 1.0);
    assert!(
        (val - 0.11490_f64).abs() < 1e-5,
        "J2(1.0) = {val}, expected about 0.11490"
    );
}

#[test]
fn bessel_j2_at_2() {
    let val = bessel_j(2, 2.0);
    assert!(
        (val - 0.35283_f64).abs() < 1e-5,
        "J2(2.0) = {val}, expected about 0.35283"
    );
}

#[test]
fn bessel_j1_odd_symmetry() {
    let x = 1.5;
    assert!(
        (bessel_j(1, -x) + bessel_j(1, x)).abs() < 1e-14,
        "J1 must be an odd function"
    );
}

#[test]
fn bessel_j0_even_symmetry() {
    let x = 2.3;
    assert!(
        (bessel_j(0, -x) - bessel_j(0, x)).abs() < 1e-14,
        "J0 must be an even function"
    );
}

#[test]
fn fubini_fundamental_undistorted_at_zero() {
    let b1 = fubini_harmonic_amplitude(1, 1e-9);
    assert!(
        (b1 - 1.0).abs() < 1e-6,
        "B1(sigma -> 0) must be 1.0, got {b1}"
    );
}

#[test]
fn fubini_harmonics_zero_at_source() {
    for n in 2..=5u32 {
        let bn = fubini_harmonic_amplitude(n, 1e-9);
        assert!(
            bn.abs() < 1e-6,
            "B_{n}(sigma -> 0) must be about 0, got {bn}"
        );
    }
}

#[test]
fn fubini_b1_at_half_shock() {
    let b1 = fubini_harmonic_amplitude(1, 0.5);
    let expected = 4.0 * bessel_j(1, 0.5);
    assert!(
        (b1 - expected).abs() < 1e-12,
        "B1(0.5) = {b1}, expected {expected}"
    );
    assert!(b1 > 0.9 && b1 < 1.0, "B1(0.5) out of expected range");
}

#[test]
fn fubini_b2_at_half_shock() {
    let b2 = fubini_harmonic_amplitude(2, 0.5);
    let expected = 2.0 * bessel_j(2, 1.0);
    assert!(
        (b2 - expected).abs() < 1e-12,
        "B2(0.5) = {b2}, expected {expected}"
    );
    assert!(b2 > 0.05 && b2 < 1.0);
}

#[test]
fn fubini_post_shock_sawtooth() {
    let b1 = fubini_harmonic_amplitude(1, 2.0);
    let expected = 2.0 / (PI * 2.0);
    assert!(
        (b1 - expected).abs() < 1e-14,
        "B1(sigma=2) = {b1}, expected {expected}"
    );
}

#[test]
fn aanonsen_fubini_harmonic_ratios() {
    let cases: &[(f64, f64)] = &[
        (0.25, bessel_j(2, 0.5) / (2.0 * bessel_j(1, 0.25))),
        (0.50, bessel_j(2, 1.0) / (2.0 * bessel_j(1, 0.50))),
        (0.75, bessel_j(2, 1.5) / (2.0 * bessel_j(1, 0.75))),
    ];

    for (sigma, expected) in cases {
        let b1 = fubini_harmonic_amplitude(1, *sigma);
        let b2 = fubini_harmonic_amplitude(2, *sigma);
        let ratio = b2 / b1;
        let rel_err = (ratio - expected).abs() / expected;
        assert!(
            rel_err < 0.01,
            "sigma = {sigma}: |P2|/|P1| = {ratio:.6}, expected {expected:.6}"
        );
    }
}

#[test]
fn burgers_returns_positive_attenuated_pressure() {
    let params = NonlinearParameters::soft_tissue();
    let p0 = 1.0e6;
    let f = 1.0e6;
    let z = 0.01;

    let p_z = burgers_equation(p0, f, z, &params);
    assert!(p_z > 0.0, "pressure must be positive");
    assert!(p_z <= p0, "attenuated pressure must not exceed source");
}

#[test]
fn burgers_at_zero_distance() {
    let params = NonlinearParameters::water();
    let p0 = 5.0e5;
    let f = 1.0e6;
    let p_z = burgers_equation(p0, f, 0.0, &params);
    assert!(
        (p_z - p0).abs() < p0 * 1e-12,
        "at z=0 burgers_equation must return P0, got {p_z}"
    );
}
