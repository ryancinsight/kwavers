//! Tests for Mie scattering implementation.
//!
//! Reference values were cross-checked against the canonical BHMIE Fortran
//! algorithm (Bohren & Huffman 1983, *Absorption and Scattering of Light by
//! Small Particles*, Appendix A) as reimplemented in `miepython`
//! (Prahl 2017, https://github.com/scottprahl/miepython, MIT-licensed),
//! whose `efficiencies_mx(m, x)` reproduces Wiscombe (1980) Table 1 to
//! machine precision.

use super::*;
use kwavers_core::constants::numerical::TWO_PI;
use eunomia::Complex64;

fn params_for(x: f64, m: Complex64) -> MieParameters {
    let wavelength = 1.0_f64;
    let radius = x * wavelength / (TWO_PI);
    MieParameters::new(radius, m, 1.0, wavelength)
}

#[test]
fn test_rayleigh_scattering() {
    let wavelength = 500e-9;
    let radius = 50e-9;
    let n_particle = Complex64::new(1.5, 0.01);
    let rayleigh = RayleighScattering::new(wavelength, radius, n_particle);
    assert!(rayleigh.scattering_cross_section() > 0.0);
    assert!(rayleigh.polarizability > 0.0);
    assert_eq!(rayleigh.depolarization_factor(), 0.0);
}

#[test]
fn test_mie_parameters() {
    let params = MieParameters::new(100e-9, Complex64::new(1.5, 0.1), 1.0, 500e-9);
    assert!(params.size_parameter() > 0.0 && params.size_parameter() < 2.0);
    assert!(params.relative_index().re > 1.0);
}

/// BHMIE / Wiscombe water sphere: x = 10, m = 1.33 + 0i  (non-absorbing dielectric).
/// Reference (miepython.efficiencies_mx):
///   Q_ext = 2.206549, Q_sca = 2.206549, Q_back = 0.561179, g = 0.712459.
#[test]
fn mie_bhmie_water_sphere_x10_m1p33() {
    let params = params_for(10.0, Complex64::new(1.33, 0.0));
    let result = MieCalculator::default().calculate(&params).unwrap();

    assert!(
        (result.extinction_efficiency - 2.206549).abs() < 1e-4,
        "Q_ext = {} (expected 2.206549)",
        result.extinction_efficiency
    );
    assert!(
        (result.scattering_efficiency - 2.206549).abs() < 1e-4,
        "Q_sca = {} (expected 2.206549)",
        result.scattering_efficiency
    );
    // Non-absorbing dielectric: Q_abs must vanish to numerical precision.
    assert!(
        result.absorption_efficiency.abs() < 1e-9,
        "Q_abs = {} (expected 0 for real m)",
        result.absorption_efficiency
    );
    assert!(
        (result.backscattering_efficiency - 0.561179).abs() < 1e-4,
        "Q_back = {} (expected 0.561179)",
        result.backscattering_efficiency
    );
    assert!(
        (result.asymmetry_parameter - 0.712459).abs() < 1e-4,
        "g = {} (expected 0.712459)",
        result.asymmetry_parameter
    );
}

/// BHMIE / Wiscombe dielectric: x = 5, m = 1.5 + 0i.
/// Reference (miepython.efficiencies_mx):
///   Q_ext = 3.927827, Q_sca = 3.927827, Q_back = 2.203881, g = 0.707295.
/// This x lies on a Mie ripple peak, so the test exercises convergence over
/// many terms with high sensitivity to the a_n / b_n formulation.
#[test]
fn mie_bhmie_dielectric_x5_m1p5() {
    let params = params_for(5.0, Complex64::new(1.5, 0.0));
    let result = MieCalculator::default().calculate(&params).unwrap();

    assert!(
        (result.extinction_efficiency - 3.927827).abs() < 1e-4,
        "Q_ext = {} (expected 3.927827)",
        result.extinction_efficiency
    );
    assert!(
        (result.scattering_efficiency - 3.927827).abs() < 1e-4,
        "Q_sca = {} (expected 3.927827)",
        result.scattering_efficiency
    );
    assert!(result.absorption_efficiency.abs() < 1e-9);
    assert!(
        (result.asymmetry_parameter - 0.707295).abs() < 1e-4,
        "g = {} (expected 0.707295)",
        result.asymmetry_parameter
    );
}

/// Mid-range size parameter: x = 1, m = 1.5 + 0i.
/// Reference (miepython.efficiencies_mx):
///   Q_ext = 0.215098, Q_sca = 0.215098, g = 0.198942.
#[test]
fn mie_bhmie_dielectric_x1_m1p5() {
    let params = params_for(1.0, Complex64::new(1.5, 0.0));
    let result = MieCalculator::default().calculate(&params).unwrap();

    assert!(
        (result.extinction_efficiency - 0.215098).abs() < 1e-4,
        "Q_ext = {} (expected 0.215098)",
        result.extinction_efficiency
    );
    assert!(
        (result.scattering_efficiency - 0.215098).abs() < 1e-4,
        "Q_sca = {} (expected 0.215098)",
        result.scattering_efficiency
    );
    assert!(result.absorption_efficiency.abs() < 1e-9);
}

/// Absorbing sphere: x = 10, m = 1.33 + 0.01 i.
/// Reference (miepython.efficiencies_mx):
///   Q_ext = 2.249241, Q_sca = 1.872112, Q_back = 0.318567, g = 0.754141.
#[test]
fn mie_absorbing_sphere_x10_m1p33_p0p01i() {
    let params = params_for(10.0, Complex64::new(1.33, 0.01));
    let result = MieCalculator::default().calculate(&params).unwrap();

    assert!(
        (result.extinction_efficiency - 2.249241).abs() < 1e-4,
        "Q_ext = {} (expected 2.249241)",
        result.extinction_efficiency
    );
    assert!(
        (result.scattering_efficiency - 1.872112).abs() < 1e-4,
        "Q_sca = {} (expected 1.872112)",
        result.scattering_efficiency
    );
    assert!(
        result.absorption_efficiency > 0.0,
        "Absorbing sphere must have Q_abs > 0"
    );
    assert!(
        ((result.extinction_efficiency - result.scattering_efficiency)
            - result.absorption_efficiency)
            .abs()
            < 1e-9,
        "Q_ext − Q_sca must equal Q_abs"
    );
}

/// Rayleigh-branch absorbing sphere: x = 0.05 routes through the small-x
/// closed-form branch. With BH convention m = n + iκ (κ ≥ 0), Q_abs must be
/// strictly positive and dominate Q_sca.
/// Reference (miepython.efficiencies_mx, x=0.05, m=1.5+0.01i):
///   Q_ext = 9.993748e-4, Q_sca = 1.442601e-6, Q_abs ≈ 9.979322e-4.
#[test]
fn mie_rayleigh_branch_absorption_sign() {
    let params = params_for(0.05, Complex64::new(1.5, 0.01));
    let result = MieCalculator::default().calculate(&params).unwrap();

    assert!(
        result.absorption_efficiency > 0.0,
        "Q_abs = {} must be positive for κ > 0",
        result.absorption_efficiency
    );
    assert!(
        (result.absorption_efficiency - 9.979322e-4).abs() / 9.979322e-4 < 5e-3,
        "Q_abs = {} (expected ~9.979e-4)",
        result.absorption_efficiency
    );
    assert!(
        result.absorption_efficiency > result.scattering_efficiency,
        "Absorption dominates scattering in the Rayleigh-absorbing limit"
    );
}

/// Rayleigh-limit cross-check: the full Mie series at x = 0.15 (above the
/// short-circuit threshold at x < 0.1) must match the closed-form Rayleigh
/// efficiency Q_sca = (8/3) x⁴ |(m²−1)/(m²+2)|² to better than 1 %.
#[test]
fn mie_rayleigh_limit_consistency() {
    let m = Complex64::new(1.5, 0.0);
    let x = 0.15_f64;
    let params = params_for(x, m);
    let result = MieCalculator::default().calculate(&params).unwrap();

    let m2 = m * m;
    let alpha = (m2 - Complex64::new(1.0, 0.0)) / (m2 + Complex64::new(2.0, 0.0));
    let q_sca_rayleigh = (8.0 / 3.0) * x.powi(4) * alpha.norm_sqr();

    let rel_err = (result.scattering_efficiency - q_sca_rayleigh).abs() / q_sca_rayleigh;
    assert!(
        rel_err < 1e-2,
        "Mie Q_sca = {}, Rayleigh Q_sca = {}, rel_err = {}",
        result.scattering_efficiency,
        q_sca_rayleigh,
        rel_err
    );
}

