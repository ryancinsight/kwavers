//! Bessel function kernel for the Fubini harmonic coefficients.

use std::f64::consts::PI;

/// Bessel function of the first kind, order `n >= 0`, argument `x`.
///
/// The implementation evaluates
///
/// ```text
/// J_n(x) = sum_{k=0}^infty (-1)^k (x/2)^(2k+n) / (k! (n+k)!)
/// ```
///
/// via the term-ratio recurrence
///
/// ```text
/// term[k+1] / term[k] = -(x/2)^2 / ((k+1)(n+k+1)).
/// ```
pub(crate) fn bessel_j(n: u32, x: f64) -> f64 {
    if x == 0.0 {
        return if n == 0 { 1.0 } else { 0.0 };
    }
    if x < 0.0 {
        return if n.is_multiple_of(2) {
            bessel_j(n, -x)
        } else {
            -bessel_j(n, -x)
        };
    }

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
        let nf = n as f64;
        (2.0 * PI * nf).sqrt() * (nf / std::f64::consts::E).powf(nf)
    }
}
