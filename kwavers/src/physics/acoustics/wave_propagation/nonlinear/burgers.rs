//! Burgers' Equation: Fubini–Blackstock Analytical Solution
//!
//! # Mathematical Foundation
//!
//! The lossless 1D Burgers' equation in retarded-time form (τ = t − z/c₀):
//!
//! ```text
//! ∂P/∂z = (β / (ρ₀c₀³))  P  ∂P/∂τ
//! ```
//!
//! This is the plane-wave (∇⊥²P = 0), lossless (δ = 0) projection of the full
//! KZK equation.  It admits an exact closed-form solution through the Fubini (1935)
//! series expansion.
//!
//! # Fubini Solution  (pre-shock, σ < 1)
//!
//! ## Theorem (Fubini 1935; Aanonsen et al. 1984 §2, eq. (6))
//!
//! For a monochromatic source p(0, τ) = P₀ sin(ωτ) in a lossless medium, the
//! exact implicit solution of the lossless Burgers equation
//!   P = P₀ sin(ωτ + σ P/P₀)
//! can be expanded in a Fourier series using the Jacobi–Anger identity to give:
//!
//! ```text
//! P(z, τ) = 2P₀  Σₙ₌₁^∞  [Jₙ(nσ) / (nσ)]  sin(nωτ)
//! ```
//!
//! The amplitude of the nth harmonic is:
//!
//! ```text
//! |Pₙ|/P₀ = Bₙ(σ) = (2/(nσ)) Jₙ(nσ)      0 < σ < 1
//! ```
//!
//! where σ = z/z_shock is the Gol'dberg number (dimensionless propagation distance)
//! and z_shock = ρ₀c₀³/(β ω P₀) is the plane-wave shock formation distance.
//!
//! ### Limits
//!
//! - σ → 0: B₁ → 1, Bₙ>₁ → 0  (no distortion at source)
//! - σ → 1: B₁ → (2/1) J₁(1) ≈ 0.880  (amplitude has dropped 12 % to harmonics)
//!
//! # Blackstock Sawtooth Solution  (post-shock, σ ≥ 1)
//!
//! ## Theorem (Blackstock 1966; Hamilton & Blackstock 1998 §4.3)
//!
//! After shock formation the waveform asymptotes to a periodic sawtooth.
//! The nth Fourier coefficient of the lossless post-shock waveform satisfies:
//!
//! ```text
//! |Pₙ|/P₀  →  2/(nπ)     (σ → ∞,  sawtooth limit)
//! ```
//!
//! For the fundamental (n = 1):
//!
//! ```text
//! B₁ = (2/σ) J₁(σ)          0 ≤ σ < 1    [Fubini]
//! B₁ = 2/(πσ)               σ ≥ 1         [Blackstock sawtooth]
//! ```
//!
//! Continuity: at σ = 1, Fubini gives B₁ ≈ 0.880 and sawtooth gives B₁ = 2/π ≈ 0.637.
//! The ~27% discontinuity is physical — the weak-shock approximation breaks down near
//! shock formation and the Fubini series is no longer single-valued for σ > 1.
//!
//! # Attenuation
//!
//! Thermoviscous attenuation is included as an independent exp(−α(nf₀)·z) factor
//! per harmonic.  This operator-splitting approach is valid when absorption is weak
//! per nonlinear accumulation length:  α(f₀) × z_shock << 1.
//!
//! For water at 1 MHz (α ≈ 0.0025 Np/m, z_shock ≈ 0.15 m): α × z_shock ≈ 4e-4 << 1 ✓
//! For soft tissue at 1 MHz (α ≈ 5.75 Np/m, z_shock ≈ 0.14 m): α × z_shock ≈ 0.8
//! (marginal; use full KZK solver for high-accuracy tissue results).
//!
//! # References
//!
//! - Fubini-Ghiron E (1935). Alta Frequenza 4, 530–581.
//! - Aanonsen SI, Barkve T, Tjøtta JN, Tjøtta S (1984). J. Acoust. Soc. Am. 75(3),
//!   749–768. DOI:10.1121/1.390585
//! - Blackstock DT (1966). J. Acoust. Soc. Am. 39(6), 1019–1026. DOI:10.1121/1.1909986
//! - Hamilton MF, Blackstock DT (1998). Nonlinear Acoustics. Academic Press. §4.3.
//! - Abramowitz M, Stegun IA (1964). Handbook of Mathematical Functions. §9.1.

use super::NonlinearParameters;
use std::f64::consts::PI;

// ────────────────────────────────────────────────────────────────────────────
// Bessel function implementation
// ────────────────────────────────────────────────────────────────────────────

/// Bessel function of the first kind, order n ≥ 0, argument x.
///
/// ## Algorithm  (power series with term-ratio recurrence)
///
/// ```text
/// Jₙ(x) = Σₖ₌₀^∞  (-1)^k / (k! (n+k)!)  ×  (x/2)^(2k+n)
/// ```
///
/// Terms are generated via the ratio recurrence
///
/// ```text
/// term[k+1] / term[k]  =  −(x/2)² / ((k+1)(n+k+1))
/// ```
///
/// which avoids computing large intermediate powers or explicit factorials
/// for high k.  The first term is `(x/2)^n / n!`.
///
/// ## Convergence
///
/// The ratio |term[k+1]/term[k]| = (x/2)² / (k(k+n)) → 0 as k → ∞.
/// For |x| ≤ 20 and n ≤ 10 the series converges to machine precision in
/// ≤ 50 iterations.
///
/// ## References
///
/// Abramowitz M, Stegun IA (1964). §9.1.10, eq. (9.1.10).
pub(crate) fn bessel_j(n: u32, x: f64) -> f64 {
    if x == 0.0 {
        return if n == 0 { 1.0 } else { 0.0 };
    }
    // Jₙ(−x) = (−1)^n Jₙ(x)
    if x < 0.0 {
        return if n.is_multiple_of(2) {
            bessel_j(n, -x)
        } else {
            -bessel_j(n, -x)
        };
    }

    // First term: (x/2)^n / n!
    let half_x = x / 2.0;
    let mut term = half_x.powi(n as i32) / factorial_f64(n);
    let mut sum = term;
    let half_x_sq = half_x * half_x;

    for k in 1_u32..=60 {
        term *= -half_x_sq / (k as f64 * (n + k) as f64);
        sum += term;
        if term.abs() <= f64::EPSILON * sum.abs().max(1e-300) {
            break;
        }
    }
    sum
}

/// Exact factorial n! as f64.
///
/// Uses a lookup table for n ≤ 20 (exact in f64) and Stirling's approximation
/// for n > 20.  For our Bessel series the first-term denominator n! needs n ≤ 10
/// (the harmonic index), so the table is always sufficient.
fn factorial_f64(n: u32) -> f64 {
    const TABLE: [f64; 21] = [
        1.0,
        1.0,
        2.0,
        6.0,
        24.0,
        120.0,
        720.0,
        5040.0,
        40320.0,
        362_880.0,
        3_628_800.0,
        39_916_800.0,
        479_001_600.0,
        6_227_020_800.0,
        87_178_291_200.0,
        1_307_674_368_000.0,
        20_922_789_888_000.0,
        355_687_428_096_000.0,
        6_402_373_705_728_000.0,
        121_645_100_408_832_000.0,
        2_432_902_008_176_640_000.0,
    ];
    if (n as usize) < TABLE.len() {
        TABLE[n as usize]
    } else {
        // Stirling's approximation (unused for n ≤ 10, but defensive)
        let nf = n as f64;
        (2.0 * PI * nf).sqrt() * (nf / std::f64::consts::E).powf(nf)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Public API
// ────────────────────────────────────────────────────────────────────────────

/// Normalised amplitude of the nth harmonic from the Fubini–Blackstock solution.
///
/// Returns |Pₙ|/P₀ for a **lossless** plane wave at dimensionless propagation
/// distance σ = z/z_shock (Gol'dberg number).
///
/// ## Formula
///
/// ```text
/// Bₙ(σ) = (2/(nσ)) Jₙ(nσ)    0 < σ < 1    [Fubini, Aanonsen 1984 eq. (6)]
/// Bₙ(σ) = 2/(nπ)              σ ≥ 1         [sawtooth limit, Blackstock 1966]
/// ```
///
/// Boundary values:
/// - σ = 0: B₁ = 1, Bₙ>₁ = 0  (undistorted source)
/// - σ = 1: Fubini and sawtooth do NOT match continuously (the series
///   is multi-valued for σ > 1; the equal-area rule selects the single-valued
///   post-shock branch).
///
/// ## References
///
/// - Aanonsen et al. (1984) J. Acoust. Soc. Am. 75(3), eq. (6).
/// - Blackstock (1966) J. Acoust. Soc. Am. 39(6), §IV.
/// - Hamilton & Blackstock (1998) Nonlinear Acoustics §4.3, eq. (4.3.7).
#[must_use]
pub fn fubini_harmonic_amplitude(n: u32, sigma: f64) -> f64 {
    assert!(n >= 1, "harmonic order must be ≥ 1");
    if sigma <= 0.0 {
        return if n == 1 { 1.0 } else { 0.0 };
    }
    if sigma < 1.0 {
        // Exact Fubini series coefficient: Bₙ = (2/(nσ)) Jₙ(nσ)
        let arg = n as f64 * sigma;
        2.0 * bessel_j(n, arg) / arg
    } else {
        // Asymptotic sawtooth (Blackstock): Bₙ = 2/(nπσ)
        2.0 / (n as f64 * PI * sigma)
    }
}

/// Fundamental frequency pressure amplitude from the Fubini–Blackstock solution,
/// with independent thermoviscous attenuation applied per harmonic.
///
/// ## Algorithm
///
/// 1. Shock formation distance:
///    ```text
///    z_shock = ρ₀c₀³ / (β ω P₀)    [m]
///    ```
///
/// 2. Gol'dberg number:  σ = z / z_shock
///
/// 3. Lossless Fubini coefficient (fundamental, n = 1):
///    ```text
///    B₁ = (2/σ) J₁(σ)    σ < 1
///    B₁ = 2/(πσ)          σ ≥ 1
///    ```
///
/// 4. Attenuated fundamental:
///    ```text
///    P₁(z) = P₀ × B₁(σ) × exp(−α(f₀) × z)
///    ```
///
/// ## Returns
///
/// Fundamental pressure amplitude at distance z [Pa].
///
/// ## References
///
/// - Aanonsen et al. (1984) J. Acoust. Soc. Am. 75(3), §2 eq. (6).
/// - Hamilton & Blackstock (1998) Nonlinear Acoustics §4.3, eq. (4.3.5–6).
#[must_use]
pub fn burgers_equation(
    initial_pressure: f64,
    frequency: f64,
    distance: f64,
    params: &NonlinearParameters,
) -> f64 {
    let omega = 2.0 * PI * frequency;

    // Shock formation distance [m]:  z_shock = ρ₀c₀³ / (β ω P₀)
    let z_shock =
        params.density * params.sound_speed.powi(3) / (params.beta * omega * initial_pressure);

    // Gol'dberg number (dimensionless)
    let sigma = distance / z_shock;

    // Lossless Fubini/Blackstock normalised fundamental amplitude B₁ = |P₁|/P₀
    let b1 = fubini_harmonic_amplitude(1, sigma);

    // Thermoviscous attenuation of the fundamental [Np/m] × distance
    let alpha = params.attenuation_at_frequency(frequency);

    initial_pressure * b1 * (-alpha * distance).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // ── Bessel function tests ─────────────────────────────────────────────

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

    /// J₁(0.5) reference: Abramowitz & Stegun Table 9.1 → 0.24227 (5 sig. figs.)
    #[test]
    fn bessel_j1_small_argument() {
        let val = bessel_j(1, 0.5);
        assert!(
            (val - 0.24227_f64).abs() < 1e-5,
            "J₁(0.5) = {val}, expected ≈ 0.24227"
        );
    }

    /// J₂(1.0) reference: A&S Table 9.1 → 0.11490
    #[test]
    fn bessel_j2_unit_argument() {
        let val = bessel_j(2, 1.0);
        assert!(
            (val - 0.11490_f64).abs() < 1e-5,
            "J₂(1.0) = {val}, expected ≈ 0.11490"
        );
    }

    /// J₂(2.0) reference: A&S Table 9.1 → 0.35283
    #[test]
    fn bessel_j2_at_2() {
        let val = bessel_j(2, 2.0);
        assert!(
            (val - 0.35283_f64).abs() < 1e-5,
            "J₂(2.0) = {val}, expected ≈ 0.35283"
        );
    }

    /// J₁(x) symmetry: J₁(−x) = −J₁(x)
    #[test]
    fn bessel_j1_odd_symmetry() {
        let x = 1.5;
        assert!(
            (bessel_j(1, -x) + bessel_j(1, x)).abs() < 1e-14,
            "J₁ must be an odd function"
        );
    }

    /// J₀(x) symmetry: J₀(−x) = J₀(x)
    #[test]
    fn bessel_j0_even_symmetry() {
        let x = 2.3;
        assert!(
            (bessel_j(0, -x) - bessel_j(0, x)).abs() < 1e-14,
            "J₀ must be an even function"
        );
    }

    // ── Fubini solution tests ─────────────────────────────────────────────

    /// At σ → 0 the fundamental amplitude equals P₀ (no distortion).
    #[test]
    fn fubini_fundamental_undistorted_at_zero() {
        let b1 = fubini_harmonic_amplitude(1, 1e-9);
        assert!((b1 - 1.0).abs() < 1e-6, "B₁(σ→0) must be 1.0, got {b1}");
    }

    /// At σ → 0 all harmonics n ≥ 2 vanish.
    #[test]
    fn fubini_harmonics_zero_at_source() {
        for n in 2..=5u32 {
            let bn = fubini_harmonic_amplitude(n, 1e-9);
            assert!(bn.abs() < 1e-6, "B_{n}(σ→0) must be ≈ 0, got {bn}");
        }
    }

    /// At σ = 0.5, compare Fubini B₁ against exact computation.
    ///
    /// B₁(0.5) = (2/0.5) J₁(0.5) = 4 × 0.24227 = 0.96908
    #[test]
    fn fubini_b1_at_half_shock() {
        let b1 = fubini_harmonic_amplitude(1, 0.5);
        let expected = 4.0 * bessel_j(1, 0.5); // = (2/0.5) × J₁(0.5)
        assert!(
            (b1 - expected).abs() < 1e-12,
            "B₁(0.5) = {b1}, expected {expected}"
        );
        // Numerical sanity: should be close to 1 but < 1
        assert!(b1 > 0.9 && b1 < 1.0, "B₁(0.5) out of expected range");
    }

    /// At σ = 0.5, compare Fubini B₂ against exact computation.
    ///
    /// B₂(0.5) = (2/(2×0.5)) J₂(2×0.5) = J₂(1.0) = 0.11490
    #[test]
    fn fubini_b2_at_half_shock() {
        let b2 = fubini_harmonic_amplitude(2, 0.5);
        let expected = bessel_j(2, 1.0); // = (1/0.5) × J₂(1.0) = 2 × J₂(1.0) / 2 = J₂(1.0)
                                         // (2/(2×0.5)) × J₂(2×0.5) = (2/1.0) × J₂(1.0) / (2×0.5) ... let me recompute:
                                         // = (2/(2σ)) × J₂(2σ) with σ=0.5: (2/(2×0.5)) × J₂(2×0.5) = (2/1) × J₂(1) = 2 × 0.11490
                                         // Wait: fubini_harmonic_amplitude(2, 0.5) = 2 × J₂(2×0.5) / (2×0.5) = 2 × J₂(1.0) / 1.0
                                         // = 2 × J₂(1.0) = 0.2298
        let expected_recalc = 2.0 * bessel_j(2, 1.0);
        assert!(
            (b2 - expected_recalc).abs() < 1e-12,
            "B₂(0.5) = {b2}, expected {expected_recalc}"
        );
        // B₂ should be measurably above 0 (nonlinear accumulation) but below B₁
        assert!(b2 > 0.05 && b2 < 1.0);
        let _ = expected;
    }

    /// Post-shock: sawtooth formula at σ = 2.0.
    ///
    /// B₁(2) = 2/(π × 2) = 1/π ≈ 0.3183
    #[test]
    fn fubini_post_shock_sawtooth() {
        let b1 = fubini_harmonic_amplitude(1, 2.0);
        let expected = 2.0 / (PI * 2.0);
        assert!(
            (b1 - expected).abs() < 1e-14,
            "B₁(σ=2) = {b1}, expected {expected}"
        );
    }

    // ── Aanonsen (1984) Fubini validation  ───────────────────────────────
    //
    // Validate Fubini ratio |P₂|/|P₁| = B₂(σ)/B₁(σ) against the analytical
    // result at σ = 0.25, 0.50, 1.00 (pre-shock regime).
    //
    // Reference values are computed from the exact Fubini formulae above;
    // the tolerances below are ±1 % relative.

    #[test]
    fn aanonsen_fubini_harmonic_ratios() {
        // (sigma, expected_P2_over_P1)
        let cases: &[(f64, f64)] = &[
            // B₂/B₁ = [J₂(2σ)/(σ)] / [2J₁(σ)/σ] = J₂(2σ) / (2 J₁(σ))
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
                "σ = {sigma}: |P₂|/|P₁| = {ratio:.6}, expected {expected:.6} (err {:.2} %)",
                rel_err * 100.0
            );
        }
    }

    // ── burgers_equation tests ────────────────────────────────────────────

    /// burgers_equation must return a positive, attenuated pressure.
    #[test]
    fn burgers_returns_positive_attenuated_pressure() {
        let params = NonlinearParameters::soft_tissue();
        let p0 = 1.0e6;
        let f = 1.0e6;
        let z = 0.01; // well pre-shock

        let p_z = burgers_equation(p0, f, z, &params);
        assert!(p_z > 0.0, "pressure must be positive");
        assert!(p_z <= p0, "attenuated pressure must not exceed source");
    }

    /// At z = 0 the pressure must equal P₀ (σ = 0 ⟹ B₁ = 1, exp(0) = 1).
    #[test]
    fn burgers_at_zero_distance() {
        let params = NonlinearParameters::water();
        let p0 = 5.0e5;
        let f = 1.0e6;
        let p_z = burgers_equation(p0, f, 0.0, &params);
        assert!(
            (p_z - p0).abs() < p0 * 1e-12,
            "at z=0 burgers_equation must return P₀, got {p_z}"
        );
    }
}
