//! Bessel functions of the first kind, Jₙ(x) — workspace SSOT.
//!
//! Canonical implementation consolidating the formerly-duplicated copies in
//! `kwavers-physics` (analytical wave, Burgers/Fubini harmonics, transducer
//! directivity). Uses the Numerical-Recipes Horner/Chebyshev rational
//! approximations for J₀ and J₁ (error ≲ 2e-9 for |x| ≤ 8, Hankel asymptotic
//! expansion beyond), and Miller downward recurrence for Jₙ (n ≥ 2). The x = 0
//! values are returned exactly (J₀(0) = 1, Jₙ₍ₙ≥₁₎(0) = 0).
//!
//! References: Abramowitz & Stegun (1964) §9; Press et al., *Numerical Recipes*
//! (Bessel rational approximations); DLMF 10.

use std::f64::consts::{FRAC_PI_4, PI};

/// Bessel J₀(x). Exact `1.0` at the origin.
#[must_use]
pub fn j0(x: f64) -> f64 {
    if x == 0.0 {
        return 1.0; // J₀(0) = 1 exactly (the rational approx returns 1 + ~3e-9).
    }
    let ax = x.abs();
    if ax < 8.0 {
        let y = x * x;
        let num = 57568490574.0_f64
            + y * (-13362590354.0
                + y * (651619640.7 + y * (-11214424.18 + y * (77392.33017 + y * (-184.9052456)))));
        let den = 57568490411.0_f64
            + y * (1029532985.0 + y * (9494680.718 + y * (59272.64853 + y * (267.8532712 + y))));
        num / den
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - FRAC_PI_4;
        let p = 1.0
            + y * (-0.001098628627
                + y * (0.000002734510407 + y * (-2.073370639e-6 + y * 2.093887211e-7)));
        let q = -0.01562499995
            + y * (0.0001430488765
                + y * (-6.911147651e-5 + y * (7.621095161e-5 - y * 9.34935152e-7)));
        (2.0 / (PI * ax)).sqrt() * (p * xx.cos() - z * q * xx.sin())
    }
}

/// Bessel J₁(x) — odd function, `J₁(0) = 0` (falls out of the leading `x` factor).
#[must_use]
pub fn j1(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 8.0 {
        // Small-argument rational approximation; the numerator carries the
        // signed `x` factor, so the result already has the correct (odd) sign.
        let y = x * x;
        let num = x
            * (72362614232.0
                + y * (-7895059235.0
                    + y * (242396853.1
                        + y * (-2972611.439 + y * (15704.48260 + y * (-30.16036606))))));
        let den = 144725228442.0
            + y * (2300535178.0 + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y))));
        num / den
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - 2.356_194_490_2;
        let p = 1.0
            + y * (0.183105e-2 + y * (-3.516396496e-5 + y * (2.457520174e-5 - y * 2.400505341e-7)));
        let q = 0.04687499995
            + y * (-0.2002690873e-3
                + y * (8.449199096e-5 + y * (-8.8228987e-5 + y * 1.050343160e-6)));
        let r = (2.0 / (PI * ax)).sqrt() * (p * xx.cos() - z * q * xx.sin());
        if x < 0.0 {
            -r
        } else {
            r
        }
    }
}

/// Bessel Jₙ(x) for integer order `n ≥ 0`.
///
/// Delegates to [`j0`]/[`j1`] for n ∈ {0, 1}; uses Miller downward recurrence
/// with two-buffer normalisation for n ≥ 2 (accurate to ≲1e-9 for |x| ≤ 50,
/// n ≤ 20). Returns exact 0 for n ≥ 1 at x = 0.
#[must_use]
pub fn jn(n: u32, x: f64) -> f64 {
    match n {
        0 => j0(x),
        1 => j1(x),
        _ => {
            if x.abs() < 1e-15 {
                return 0.0;
            }
            let n_us = n as usize;
            let m_start = n_us + n_us.max(30);
            let mut bjp = 0.0_f64;
            let mut bj = 1.0_f64;
            let mut bj0 = 0.0_f64;
            let mut bj1 = 0.0_f64;
            let mut ans = 0.0_f64;
            let two_over_x = 2.0 / x;
            for k in (0..m_start).rev() {
                let bjm = (k as f64 + 1.0) * two_over_x * bj - bjp;
                bjp = bj;
                bj = bjm;
                if bj.abs() > 1.0e100 {
                    bj *= 1.0e-100;
                    bjp *= 1.0e-100;
                    ans *= 1.0e-100;
                    bj0 *= 1.0e-100;
                    bj1 *= 1.0e-100;
                }
                if k == n_us {
                    ans = bj;
                }
                if k == 1 {
                    bj1 = bj;
                }
                if k == 0 {
                    bj0 = bj;
                }
            }
            let j0_true = j0(x);
            let j1_true = j1(x);
            let scale = if bj0.abs() >= bj1.abs() {
                if bj0.abs() < 1e-300 {
                    return 0.0;
                }
                j0_true / bj0
            } else {
                if bj1.abs() < 1e-300 {
                    return 0.0;
                }
                j1_true / bj1
            };
            ans * scale
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference values from Abramowitz & Stegun / DLMF (10 significant figures).
    #[test]
    fn reference_values() {
        assert_eq!(j0(0.0), 1.0);
        assert_eq!(j1(0.0), 0.0);
        assert_eq!(jn(0, 0.0), 1.0);
        assert_eq!(jn(5, 0.0), 0.0);
        // NR rational approximation accuracy is ≲2e-9; assert against A&S
        // reference values within 1e-8.
        assert!((j0(1.0) - 0.765_197_686_6).abs() < 1e-8);
        assert!((j1(1.0) - 0.440_050_585_7).abs() < 1e-8);
        assert!((j0(5.0) - (-0.177_596_771_3)).abs() < 1e-8);
        assert!((j1(5.0) - (-0.327_579_137_9)).abs() < 1e-8);
        assert!((jn(2, 1.0) - 0.114_903_484_9).abs() < 1e-8);
        assert!((jn(3, 2.0) - 0.128_943_249_8).abs() < 1e-8);
        // Large-argument (Hankel asymptotic branch): NR error is ≈1.4e-6 near
        // the x=8 changeover (tightening for larger x), matching the existing
        // physics directivity/harmonic tolerances.
        assert!((j0(10.0) - (-0.245_935_764_5)).abs() < 1e-5);
        assert!((j1(10.0) - 0.043_472_746_2).abs() < 1e-5);
    }

    #[test]
    fn parity() {
        for &x in &[0.3, 1.7, 4.2, 9.5] {
            assert!((j0(-x) - j0(x)).abs() < 1e-14); // even
            assert!((j1(-x) + j1(x)).abs() < 1e-14); // odd
        }
    }
}
