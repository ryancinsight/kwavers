//! Fubini-Blackstock analytical Burgers solution.

use super::bessel_j;
use crate::acoustics::wave_propagation::nonlinear::NonlinearParameters;
use kwavers_core::constants::numerical::TWO_PI;

/// Normalized amplitude of the nth harmonic from the Fubini-Blackstock solution.
///
/// Returns `|P_n|/P0` for a lossless plane wave at dimensionless propagation
/// distance `sigma = z / z_shock`.
///
/// ```text
/// B_n(sigma) = (2/(n sigma)) J_n(n sigma)    0 < sigma < 1   (Fubini, pre-shock)
/// B_n(sigma) = 2/(n (1 + sigma))             sigma >= 1      (Fay sawtooth)
/// ```
/// The post-shock branch is the Fay sawtooth solution (Blackstock 1966): every
/// harmonic falls as `1/n` and the spectrum decays as `1/(1+sigma)`. At the
/// shock (`sigma = 1`) the sawtooth fundamental is `B_1 = 1` (full source
/// amplitude `P_0`), whereas the Fubini fundamental there is depleted to
/// `2 J_1(1) ≈ 0.88 P_0` — the two solutions are connected through the
/// transition region `1 ≲ sigma ≲ 3`.
///
/// # Panics
/// - Panics if assertion fails: `harmonic order must be >= 1`.
///
#[must_use]
pub fn fubini_harmonic_amplitude(n: u32, sigma: f64) -> f64 {
    assert!(n >= 1, "harmonic order must be >= 1");
    if sigma <= 0.0 {
        if n == 1 {
            1.0
        } else {
            0.0
        }
    } else if sigma < 1.0 {
        // Pre-shock: Fubini Bessel series.
        let arg = n as f64 * sigma;
        2.0 * bessel_j(n, arg) / arg
    } else {
        // Post-shock: Fay sawtooth, B_n = 2/(n(1+sigma)) (Blackstock 1966);
        // matches analytical::wave::sawtooth_harmonic_amplitude.
        2.0 / (n as f64 * (1.0 + sigma))
    }
}

/// Fundamental pressure amplitude from the Fubini-Blackstock solution with
/// independent thermoviscous attenuation applied to the fundamental.
///
/// ```text
/// z_shock = rho0 c0^3 / (beta omega P0)
/// sigma = z / z_shock
/// P1(z) = P0 B1(sigma) exp(-alpha(f0) z)
/// ```
#[must_use]
pub fn burgers_equation(
    initial_pressure: f64,
    frequency: f64,
    distance: f64,
    params: &NonlinearParameters,
) -> f64 {
    let omega = TWO_PI * frequency;
    let z_shock =
        params.density * params.sound_speed.powi(3) / (params.beta * omega * initial_pressure);
    let sigma = distance / z_shock;
    let b1 = fubini_harmonic_amplitude(1, sigma);
    let alpha = params.attenuation_at_frequency(frequency);

    initial_pressure * b1 * (-alpha * distance).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fubini_source_and_fundamental() {
        // At the source (sigma = 0) the fundamental carries the full amplitude
        // and the harmonics vanish.
        assert!((fubini_harmonic_amplitude(1, 0.0) - 1.0).abs() < 1e-12);
        assert_eq!(fubini_harmonic_amplitude(2, 0.0), 0.0);
        // Pre-shock fundamental at sigma → 1⁻ is the Fubini value 2·J_1(1) ≈ 0.880.
        let b1_pre = fubini_harmonic_amplitude(1, 0.999_999);
        assert!((b1_pre - 0.880).abs() < 2e-3, "Fubini B1(1⁻) = {b1_pre}");
    }

    #[test]
    fn fay_sawtooth_starts_at_full_amplitude_at_shock() {
        // Regression: the post-shock branch was 2/(nπσ), giving B1(1)=0.637 and
        // contradicting the physics. The Fay sawtooth (Blackstock 1966) is
        // B_n = 2/(n(1+σ)); at the shock σ=1 the sawtooth fundamental is the
        // full source amplitude B1 = 2/(1·2) = 1.
        assert!((fubini_harmonic_amplitude(1, 1.0) - 1.0).abs() < 1e-12);
        // Every harmonic falls as 1/n, the spectrum as 1/(1+σ).
        for &(n, sigma, expected) in &[
            (1u32, 1.0, 1.0),
            (2, 1.0, 0.5),
            (1, 3.0, 0.5),
            (3, 3.0, 1.0 / 6.0),
        ] {
            let b = fubini_harmonic_amplitude(n, sigma);
            assert!(
                (b - expected).abs() < 1e-12,
                "B_{n}({sigma}) = {b} != {expected}"
            );
        }
        // Old-age decay: B1 monotonically decreasing for σ ≥ 1.
        assert!(fubini_harmonic_amplitude(1, 5.0) < fubini_harmonic_amplitude(1, 1.0));
    }
}
