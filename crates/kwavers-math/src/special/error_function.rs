//! Gaussian error function — part of the `kwavers-math::special` SSOT.
//!
//! Pure, domain-free numerics depended on by the physics and clinical layers.

/// Gaussian error function `erf(x)` via the Abramowitz & Stegun 7.1.26 rational
/// approximation.
///
/// ```text
/// erf(x) ≈ sign(x) · [1 − (a₁t + a₂t² + a₃t³ + a₄t⁴ + a₅t⁵)·e^(−x²)],
///          t = 1/(1 + p·|x|)
/// ```
///
/// Odd by construction (`erf(−x) = −erf(x)`), `erf(0) = 0`, and
/// `erf(x) → ±1` as `x → ±∞`.
///
/// # Accuracy
/// Maximum absolute error `|ε| ≤ 1.5×10⁻⁷` over all real `x`
/// (Abramowitz & Stegun 1964, §7.1.26).
///
/// # Reference
/// Abramowitz, M. & Stegun, I. A. (1964). *Handbook of Mathematical Functions*,
/// §7.1.26. National Bureau of Standards.
#[inline]
#[must_use]
pub fn erf(x: f64) -> f64 {
    /// A&S 7.1.26 scale parameter `p`.
    const P: f64 = 0.327_591_1;
    /// A&S 7.1.26 polynomial coefficients `a₁..a₅` (Horner order low→high).
    const A: [f64; 5] = [
        0.254_829_592,
        -0.284_496_736,
        1.421_413_741,
        -1.453_152_027,
        1.061_405_429,
    ];

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + P * ax);
    // Horner evaluation of t·(a₁ + t(a₂ + t(a₃ + t(a₄ + t·a₅)))).
    let poly = (A[4]
        .mul_add(t, A[3])
        .mul_add(t, A[2])
        .mul_add(t, A[1])
        .mul_add(t, A[0]))
        * t;
    sign * poly.mul_add(-(-ax * ax).exp(), 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use eunomia::assert_abs_diff_eq;

    /// The A&S 7.1.26 error bound.
    const ERF_AS_TOL: f64 = 1.5e-7;

    #[test]
    fn erf_matches_reference_values() {
        // Reference values from high-precision erf (e.g. Wolfram / libm), to the
        // A&S 7.1.26 documented accuracy.
        assert_abs_diff_eq!(erf(0.0), 0.0, epsilon = ERF_AS_TOL);
        assert_abs_diff_eq!(erf(0.5), 0.520_499_877_8, epsilon = ERF_AS_TOL);
        assert_abs_diff_eq!(erf(1.0), 0.842_700_792_9, epsilon = ERF_AS_TOL);
        assert_abs_diff_eq!(erf(2.0), 0.995_322_265_0, epsilon = ERF_AS_TOL);
    }

    #[test]
    fn erf_is_odd() {
        for &x in &[0.1, 0.5, 1.0, 2.5, 4.0] {
            assert_abs_diff_eq!(erf(-x), -erf(x), epsilon = 1e-15);
        }
    }

    #[test]
    fn erf_saturates_to_unit_magnitude() {
        assert_abs_diff_eq!(erf(6.0), 1.0, epsilon = ERF_AS_TOL);
        assert_abs_diff_eq!(erf(-6.0), -1.0, epsilon = ERF_AS_TOL);
        // Monotone non-decreasing across the transition.
        assert!(erf(0.5) < erf(1.0) && erf(1.0) < erf(2.0));
    }
}
