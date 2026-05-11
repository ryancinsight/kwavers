use super::*;
use crate::physics::optics::quantum_optics::constants::{C, E_CHARGE, KB};
use std::f64::consts::PI;

/// Lyman-alpha: lambda = 121.567 nm, f12 = 0.4162, g1 = 2, g2 = 6.
/// Reference A21 = 6.265e8 s^-1.
/// # Panics
/// - Panics if assertion fails: `Lyman-alpha A21: got {:.4e}, expected {:.4e} (err {:.1}%)`.
///
#[test]
fn test_einstein_a21_hydrogen_lyman_alpha() {
    let lambda = 121.567e-9;
    let omega21 = 2.0 * PI * C / lambda;
    let coeff = EinsteinCoefficients::from_oscillator_strength(omega21, 0.4162, 2.0, 6.0);
    let expected = 6.265e8;
    let rel_err = (coeff.a21 - expected).abs() / expected;
    assert!(
        rel_err < 0.05,
        "Lyman-alpha A21: got {:.4e}, expected {:.4e} (err {:.1}%)",
        coeff.a21,
        expected,
        100.0 * rel_err
    );
}

/// Radiative lifetime for hydrogen Lyman-alpha must be near 1.6 ns.
/// # Panics
/// - Panics if assertion fails: `Lyman-alpha lifetime must be 1-3 ns, got {tau:.3e} s`.
///
#[test]
fn test_radiative_lifetime_lyman_alpha() {
    let omega21 = 2.0 * PI * C / 121.567e-9;
    let coeff = EinsteinCoefficients::from_oscillator_strength(omega21, 0.4162, 2.0, 6.0);
    let tau = coeff.radiative_lifetime();
    assert!(
        tau > 1.0e-9 && tau < 3.0e-9,
        "Lyman-alpha lifetime must be 1-3 ns, got {tau:.3e} s"
    );
}

/// Einstein detailed balance requires B12 / B21 = g2 / g1.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_einstein_b_degeneracy_relation() {
    let omega = 2.0 * PI * C / 400e-9;
    let coeff = EinsteinCoefficients::from_oscillator_strength(omega, 0.5, 1.0, 3.0);
    let ratio = coeff.b12 / coeff.b21;
    assert!(
        (ratio - 3.0).abs() < 1e-13,
        "B12/B21 = g2/g1 = 3, got {ratio:.10}"
    );
}

/// Invalid oscillator-strength inputs are non-finite, not silently regularized.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_einstein_invalid_degeneracy_is_nonfinite() {
    let coeff = EinsteinCoefficients::from_oscillator_strength(1.0e15, 0.5, 1.0, 0.0);
    assert!(coeff.a21.is_nan());
    assert!(coeff.b12.is_nan());
}

/// Flash emission fraction follows the short-time Poisson expansion.
/// # Panics
/// - Panics if assertion fails: `Flash emission fraction must be < 30% for short flash`.
/// - Panics if assertion fails: `Flash fraction must approach A21*dt for short flash`.
///
#[test]
fn test_flash_emission_fraction_sbsl() {
    let omega = 2.0 * PI * C / 300e-9;
    let coeff = EinsteinCoefficients::from_oscillator_strength(omega, 0.4, 1.0, 1.0);
    let flash_dt = 100e-12;
    let frac = coeff.flash_emission_fraction(flash_dt);
    assert!(
        frac > 0.0 && frac < 0.3,
        "Flash emission fraction must be < 30% for short flash"
    );
    assert!(
        frac < coeff.a21 * flash_dt * 1.05,
        "Flash fraction must approach A21*dt for short flash"
    );
}

/// Gaunt factor at SBSL visible conditions must lie in the published range.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_gaunt_factor_sbsl_conditions() {
    let nu = C / 400e-9;
    let g_ff = gaunt_factor_ff(nu, 10_000.0, 1.0);
    assert!(
        (0.3..=1.5).contains(&g_ff),
        "Gaunt factor at SBSL conditions must be in [0.3, 1.5], got {g_ff:.4}"
    );
}

/// Gaunt factor decreases from red to UV wavelengths at fixed SBSL temperature.
/// # Panics
/// - Panics if assertion fails: `Gaunt factor at 800 nm ({g_red:.4}) must exceed 200 nm ({g_uv:.4})`.
///
#[test]
fn test_gaunt_factor_frequency_dependence() {
    let g_red = gaunt_factor_ff(C / 800e-9, 10_000.0, 1.0);
    let g_uv = gaunt_factor_ff(C / 200e-9, 10_000.0, 1.0);
    assert!(
        g_red > g_uv,
        "Gaunt factor at 800 nm ({g_red:.4}) must exceed 200 nm ({g_uv:.4})"
    );
}

/// Invalid Gaunt-factor inputs produce NaN instead of a constant placeholder.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_gaunt_factor_invalid_domain_is_nan() {
    assert!(gaunt_factor_ff(0.0, 10_000.0, 1.0).is_nan());
    assert!(gaunt_factor_ff(C / 400e-9, 0.0, 1.0).is_nan());
}

/// Relativistic parameter at SBSL temperature must be much smaller than 1e-3.
/// # Panics
/// - Panics if assertion fails: `At 15 000 K, relativistic parameter = {rel:.3e} must be << 1e-3`.
///
#[test]
fn test_relativistic_parameter_sbsl() {
    let rel = relativistic_parameter(15_000.0);
    assert!(
        rel < 1e-3,
        "At 15 000 K, relativistic parameter = {rel:.3e} must be << 1e-3"
    );
}

/// Lamb shift ratio must be negligible at SBSL temperatures.
/// # Panics
/// - Panics if assertion fails: `Lamb shift / kT at 10 000 K = {ratio:.3e} must be < 1e-4`.
///
#[test]
fn test_lamb_shift_negligible_at_sbsl_temperature() {
    let k_t_ev = KB * 10_000.0 / E_CHARGE;
    let ratio = lamb_shift_ev(1.0) / k_t_ev;
    assert!(
        ratio < 1e-4,
        "Lamb shift / kT at 10 000 K = {ratio:.3e} must be < 1e-4"
    );
}

/// Classical bremsstrahlung is adequate for SBSL thermal conditions.
/// # Panics
/// - Panics if assertion fails: `Classical accuracy at 10 000 K: {:.6}%`.
///
#[test]
fn test_classical_bremsstrahlung_adequate_sbsl() {
    let omega_uv = 2.0 * PI * C / 300e-9;
    let assessment = QuantumCorrectionAssessment::assess(10_000.0, 100e-12, omega_uv, 0.4);
    assert!(
        assessment.classical_bremsstrahlung_adequate(),
        "Classical bremsstrahlung must be adequate at 10 000 K"
    );
    assert!(
        assessment.lamb_shift_negligible(),
        "Lamb shift must be negligible at 10 000 K"
    );
    assert!(
        assessment.classical_accuracy_pct > 99.99,
        "Classical accuracy at 10 000 K: {:.6}%",
        assessment.classical_accuracy_pct
    );
}
