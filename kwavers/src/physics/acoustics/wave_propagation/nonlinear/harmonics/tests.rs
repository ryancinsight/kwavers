use super::*;
use crate::core::constants::numerical::{MHZ_TO_HZ, MPA_TO_PA};
use crate::physics::acoustics::wave_propagation::nonlinear::{
    NonlinearParameters, TissueHarmonicProperties,
};

fn water() -> NonlinearParameters {
    NonlinearParameters::water()
}

fn soft_tissue() -> NonlinearParameters {
    NonlinearParameters::soft_tissue()
}

#[test]
fn second_harmonic_positive_and_bounded() {
    let params = soft_tissue();
    let p0 = 2.0 * MPA_TO_PA;
    let f = 2.0 * MHZ_TO_HZ;
    let z = 0.05;
    let p2 = second_harmonic_amplitude(p0, f, z, &params);
    assert!(p2 > 0.0, "P₂ must be positive");
    assert!(p2 < p0, "P₂ must be less than P₀");
}

#[test]
fn second_harmonic_measurable_at_5cm() {
    let params = soft_tissue();
    let p0 = 2.0 * MPA_TO_PA;
    let f = 2.0 * MHZ_TO_HZ;
    let z = 0.05;
    let p2 = second_harmonic_amplitude(p0, f, z, &params);
    assert!(
        p2 > p0 * 0.01,
        "P₂({z} m) = {p2:.1} Pa should exceed 1 % of P₀"
    );
}

#[test]
fn second_harmonic_zero_at_source() {
    let params = water();
    let p2 = second_harmonic_amplitude(MPA_TO_PA, MHZ_TO_HZ, 0.0, &params);
    assert!(p2.abs() < 1e-6, "P₂(z=0) must be 0, got {p2}");
}

#[test]
fn harmonic_order_hierarchy() {
    let params = water();
    let p0 = 5.0e5;
    let f = MHZ_TO_HZ;
    let z = 0.02;
    let p1 = nth_harmonic_amplitude(p0, f, z, 1, &params);
    let p2 = nth_harmonic_amplitude(p0, f, z, 2, &params);
    let p3 = nth_harmonic_amplitude(p0, f, z, 3, &params);
    assert!(p1 > p2, "P₁ > P₂ must hold at pre-shock distance");
    assert!(p2 > p3, "P₂ > P₃ must hold at pre-shock distance");
}

#[test]
fn aanonsen_1984_fubini_p2_over_p1_ratios() {
    use crate::physics::acoustics::wave_propagation::nonlinear::burgers::{
        bessel_j, fubini_harmonic_amplitude,
    };

    for &sigma in &[0.25, 0.50, 0.75] {
        let b1 = fubini_harmonic_amplitude(1, sigma);
        let b2 = fubini_harmonic_amplitude(2, sigma);
        let expected_ratio = b2 / b1;

        let analytic = bessel_j(2, 2.0 * sigma) / (2.0 * bessel_j(1, sigma));
        let discrepancy = (expected_ratio - analytic).abs() / analytic;
        assert!(
            discrepancy < 1e-12,
            "σ={sigma}: fubini ratio {expected_ratio} vs analytic {analytic}, discrepancy {discrepancy}"
        );

        assert!(
            expected_ratio > 0.0 && expected_ratio < 1.0,
            "σ={sigma}: B₂/B₁ = {expected_ratio} must be in (0,1)"
        );
    }
}

#[test]
fn thi_efficiency_maximum_at_optimal_depth() {
    let params = soft_tissue();
    let f0 = 2.0 * MHZ_TO_HZ;
    let alpha1 = params.attenuation_at_frequency(f0);
    let alpha2 = params.attenuation_at_frequency(2.0 * f0);
    let kappa = alpha1 + alpha2;
    let f_opt = 1.0 / kappa;

    let props = TissueHarmonicProperties {
        fundamental_frequency: f0,
        fundamental_pressure: MPA_TO_PA,
        fractional_bandwidth: 0.6,
        f_number: 2.0,
        focal_depth: f_opt,
    };

    let eta = tissue_harmonic_efficiency(&props, &params);
    assert!((eta - 1.0).abs() < 1e-10, "η(F_opt) must be 1.0, got {eta}");
}

#[test]
fn thi_efficiency_less_than_one_away_from_optimum() {
    let params = soft_tissue();
    let f0 = 2.0 * MHZ_TO_HZ;
    let alpha1 = params.attenuation_at_frequency(f0);
    let alpha2 = params.attenuation_at_frequency(2.0 * f0);
    let kappa = alpha1 + alpha2;
    let f_opt = 1.0 / kappa;

    for scale in [0.3, 0.5, 2.0, 5.0] {
        let props = TissueHarmonicProperties {
            fundamental_frequency: f0,
            fundamental_pressure: MPA_TO_PA,
            fractional_bandwidth: 0.6,
            f_number: 2.0,
            focal_depth: f_opt * scale,
        };
        let eta = tissue_harmonic_efficiency(&props, &params);
        assert!(
            eta > 0.0 && eta < 1.0,
            "η(F = {scale}×F_opt) = {eta} must be in (0,1)"
        );
    }
}

#[test]
fn optimal_frequency_in_medical_range() {
    let params = soft_tissue();
    let f_opt = optimal_harmonic_frequency(0.05, &params);
    assert!(
        (MHZ_TO_HZ..=15.0 * MHZ_TO_HZ).contains(&f_opt),
        "f_opt = {f_opt:.2e} Hz must be in [1, 15] MHz"
    );
}

#[test]
fn contrast_response_positive_finite() {
    let r = contrast_harmonic_response(1.0e5, 2.0 * MHZ_TO_HZ, 3.0 * MHZ_TO_HZ);
    assert!(r > 0.0 && r.is_finite());
}
