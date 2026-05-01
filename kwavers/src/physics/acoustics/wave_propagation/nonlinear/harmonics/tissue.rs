use super::super::{NonlinearParameters, TissueHarmonicProperties};

/// Normalised THI efficiency at focal depth F, dimensionless ∈ (0, 1].
///
/// ```text
/// η(F) = e × κ × F × exp(−κ × F), κ = α(f₀) + α(2f₀)
/// ```
///
/// ## Theorem
/// `F exp(−κF)` is maximised at `F = 1/κ`; normalising by the maximum gives
/// the expression above.
#[must_use]
pub fn tissue_harmonic_efficiency(
    props: &TissueHarmonicProperties,
    params: &NonlinearParameters,
) -> f64 {
    let alpha1 = params.attenuation_at_frequency(props.fundamental_frequency);
    let alpha2 = params.attenuation_at_frequency(2.0 * props.fundamental_frequency);
    let kappa = alpha1 + alpha2;

    if kappa <= 0.0 {
        return 1.0;
    }

    let f = props.focal_depth;
    std::f64::consts::E * kappa * f * (-kappa * f).exp()
}

/// Optimal fundamental frequency for maximum second-harmonic return at given depth [Hz].
///
/// ```text
/// f_opt = [α₀(1 + 2^y) y F]^{−1/y}
/// ```
///
/// `params.attenuation_coeff` is in Np/m/MHz^y and is converted to Np/m/Hz^y.
#[must_use]
pub fn optimal_harmonic_frequency(depth: f64, params: &NonlinearParameters) -> f64 {
    let y = params.attenuation_exponent;
    let alpha0_per_hz_y = params.attenuation_coeff / (1.0e6_f64).powf(y);
    let c = alpha0_per_hz_y * (1.0 + 2.0_f64.powf(y)) * y * depth;

    if c <= 0.0 {
        return 2.0e6;
    }

    (1.0 / c).powf(1.0 / y).clamp(1.0e6, 15.0e6)
}
