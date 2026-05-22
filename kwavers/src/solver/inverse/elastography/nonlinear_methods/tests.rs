use super::super::config::NonlinearInversionConfig;
use super::bayesian::bayesian_inversion;
use super::harmonic_ratio::harmonic_ratio_inversion;
use super::helpers::{
    a_landau, ba_from_beta_s, beta_s_from_amplitudes, forward_model, forward_model_derivative,
    shear_modulus,
};
use super::least_squares::nonlinear_least_squares_inversion;
use super::processor::NonlinearInversion;
use crate::domain::imaging::ultrasound::elastography::NonlinearInversionMethod;
use crate::physics::acoustics::imaging::modalities::elastography::HarmonicDisplacementField;
use std::f64::consts::PI;

fn test_config() -> NonlinearInversionConfig {
    NonlinearInversionConfig::new(NonlinearInversionMethod::HarmonicRatio)
        .with_shear_properties(3.0, 100.0, 0.05)
}

/// β_s → A₂ → β_s round-trip must return input β_s to within rtol=1e-10.
///
/// Forward: A₂ = β_s k_s A₁² z / 2
/// Inverse: β_s_rec = 2 A₂ c_s / (ω A₁² z)   [Rénier 2008, Eq. 8]
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_beta_s_round_trip() {
    let config = test_config();
    let omega = 2.0 * PI * config.excitation_frequency;
    let c_s = config.shear_wave_speed;
    let k_s = omega / c_s;
    let z = config.propagation_distance;

    for &ba_target in &[1.0_f64, 5.0, 10.0, 15.0] {
        let beta_target = ba_target / 2.0 + 1.0;
        let a1: f64 = 1e-6;
        let a2 = beta_target * k_s * a1 * a1 * z / 2.0;

        let beta_rec = beta_s_from_amplitudes(a1, a2, &config).unwrap();
        let rel_err = (beta_rec - beta_target).abs() / beta_target.abs().max(1e-12);
        assert!(
            rel_err < 1e-10,
            "Round-trip failed for B/A={ba_target}: got β_s={beta_rec:.6}, \
             expected {beta_target:.6}, rel_err={rel_err:.2e}"
        );
    }
}

/// ba_from_beta_s at β_s=1.0 (linear medium) must give B/A=0.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_ba_zero_for_linear_medium() {
    let ba = ba_from_beta_s(1.0);
    assert!(
        ba.abs() < 1e-15,
        "Linear medium (β_s=1) must give B/A=0, got {ba}"
    );
}

/// A_L sign check (Destrade & Ogden 2010, Eq. 3.8).
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_a_landau_sign() {
    let mu = 9000.0;
    assert!(a_landau(mu, 2.0) > 0.0, "A_L should be positive for β_s=2");
    assert!(
        a_landau(mu, 0.5) < 0.0,
        "A_L should be negative for β_s=0.5"
    );
}

/// For gelatin phantom: β_s ≈ 1.8, recover B/A ≈ 1.6 within 0.1%.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_gelatin_phantom_reference_value() {
    let config = test_config();
    let omega = 2.0 * PI * config.excitation_frequency;
    let c_s = config.shear_wave_speed;
    let k_s = omega / c_s;
    let z = config.propagation_distance;

    let beta_ref = 1.8_f64;
    let a1 = 1e-6_f64;
    let a2 = beta_ref * k_s * a1 * a1 * z / 2.0;

    let beta_rec = beta_s_from_amplitudes(a1, a2, &config).unwrap();
    let ba_rec = ba_from_beta_s(beta_rec);
    let ba_expected = ba_from_beta_s(beta_ref);

    let err = (ba_rec - ba_expected).abs();
    assert!(
        err < 1e-3 * ba_expected.abs().max(1.0),
        "Gelatin phantom B/A: got {ba_rec:.4}, expected {ba_expected:.4}, err={err:.2e}"
    );
}

/// forward_model + beta_s_from_amplitudes must be inverses.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_forward_model_invertible() {
    let config = test_config();
    let a1 = 2e-6;
    let ba_in = 6.0_f64;

    let (_a1_pred, a2_pred) = forward_model(ba_in, a1, &config);
    let beta_rec = beta_s_from_amplitudes(a1, a2_pred, &config).unwrap();
    let ba_rec = ba_from_beta_s(beta_rec).clamp(0.0, 20.0);

    let err = (ba_rec - ba_in).abs();
    assert!(
        err < 1e-9,
        "forward_model/beta_s_from_amplitudes inverse: got {ba_rec:.6}, \
         expected {ba_in:.6}, err={err:.2e}"
    );
}

/// Jacobian: numerical ≈ analytical to 0.1%.
/// # Panics
/// - Panics if the relative error between numerical and analytical Jacobians exceeds tolerance.
///
#[test]
fn test_forward_model_jacobian_numerical() {
    let config = test_config();
    let a1 = 1e-6_f64;
    let ba0 = 5.0_f64;
    let h = 1e-4;

    let (_, a2_pos) = forward_model(ba0 + h, a1, &config);
    let (_, a2_neg) = forward_model(ba0 - h, a1, &config);
    let da2_numerical = (a2_pos - a2_neg) / (2.0 * h);

    let (_, da2_analytical) = forward_model_derivative(ba0, a1, &config);

    let rel_err = (da2_numerical - da2_analytical).abs() / da2_analytical.abs().max(1e-30);
    assert!(
        rel_err < 1e-3,
        "Jacobian rel_err={rel_err:.2e}: numerical={da2_numerical:.4e}, \
         analytical={da2_analytical:.4e}"
    );
}

#[test]
fn test_harmonic_ratio_inversion() {
    let grid = crate::domain::grid::Grid::new(10, 10, 10, 0.001, 0.001, 0.001).unwrap();
    let harmonic_field = HarmonicDisplacementField::new(10, 10, 10, 2, 10);

    let config = NonlinearInversionConfig::new(NonlinearInversionMethod::HarmonicRatio);
    let map = harmonic_ratio_inversion(&harmonic_field, &grid, &config).unwrap();
    assert_eq!(
        map.nonlinearity_parameter.dim(),
        (10, 10, 10),
        "harmonic ratio output must span the full 10×10×10 grid"
    );
}

#[test]
fn test_nonlinear_least_squares_inversion() {
    let grid = crate::domain::grid::Grid::new(10, 10, 10, 0.001, 0.001, 0.001).unwrap();
    let harmonic_field = HarmonicDisplacementField::new(10, 10, 10, 2, 10);

    let config = NonlinearInversionConfig::new(NonlinearInversionMethod::NonlinearLeastSquares);
    let map = nonlinear_least_squares_inversion(&harmonic_field, &grid, &config).unwrap();
    assert_eq!(
        map.nonlinearity_parameter.dim(),
        (10, 10, 10),
        "nonlinear least squares output must span the full 10×10×10 grid"
    );
}

#[test]
fn test_bayesian_inversion() {
    let grid = crate::domain::grid::Grid::new(10, 10, 10, 0.001, 0.001, 0.001).unwrap();
    let harmonic_field = HarmonicDisplacementField::new(10, 10, 10, 2, 10);

    let config = NonlinearInversionConfig::new(NonlinearInversionMethod::BayesianInversion);
    let map = bayesian_inversion(&harmonic_field, &grid, &config).unwrap();
    assert_eq!(
        map.nonlinearity_parameter.dim(),
        (10, 10, 10),
        "bayesian inversion output must span the full 10×10×10 grid"
    );
}

#[test]
fn test_all_nonlinear_methods() {
    let grid = crate::domain::grid::Grid::new(10, 10, 10, 0.001, 0.001, 0.001).unwrap();
    let harmonic_field = HarmonicDisplacementField::new(10, 10, 10, 2, 10);

    for method in [
        NonlinearInversionMethod::HarmonicRatio,
        NonlinearInversionMethod::NonlinearLeastSquares,
        NonlinearInversionMethod::BayesianInversion,
    ] {
        let config = NonlinearInversionConfig::new(method);
        let processor = NonlinearInversion::new(config);
        let map = processor
            .reconstruct(&harmonic_field, &grid)
            .unwrap_or_else(|e| panic!("Nonlinear method {method:?} should succeed; got: {e:?}"));
        assert_eq!(
            map.nonlinearity_parameter.dim(),
            (10, 10, 10),
            "method {method:?}: nonlinearity_parameter must span 10×10×10 grid"
        );
    }
}

#[test]
fn test_nonlinear_inversion_processor() {
    let config = NonlinearInversionConfig::new(NonlinearInversionMethod::HarmonicRatio);
    let processor = NonlinearInversion::new(config);

    assert_eq!(processor.method(), NonlinearInversionMethod::HarmonicRatio);
    assert_eq!(processor.config().density, 1000.0);
    assert_eq!(processor.config().acoustic_speed, crate::core::constants::fundamental::SOUND_SPEED_TISSUE);
    assert_eq!(processor.config().shear_wave_speed, 3.0);
    assert_eq!(processor.config().excitation_frequency, 100.0);
    assert_eq!(processor.config().propagation_distance, 0.05);
}

/// shear_modulus: μ = ρ c_s² for default config (ρ=1000, c_s=3) → 9000 Pa.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_shear_modulus_default() {
    let config = NonlinearInversionConfig::default();
    let mu = shear_modulus(&config);
    let expected = 1000.0 * 3.0 * 3.0;
    assert!(
        (mu - expected).abs() < 1e-10,
        "shear_modulus should be {expected}, got {mu}"
    );
}
