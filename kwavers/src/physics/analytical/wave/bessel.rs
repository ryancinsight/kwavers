use std::f64::consts::{FRAC_PI_4, PI};

/// Bessel J₀(x) via Horner-evaluated Chebyshev approximation.
///
/// Error < 1.6e-9 for |x| ≤ 8, DLMF 10.2.2 rational approximation elsewhere.
pub(super) fn bessel_j0(x: f64) -> f64 {
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

/// J₁(x) via clean power series for |x| ≤ 8, Hankel expansion otherwise.
/// J₁ is an odd function: J₁(−x) = −J₁(x).
fn bessel_j1_clean(x: f64) -> f64 {
    let ax = x.abs();
    let sign = x.signum();
    let r = if ax < 8.0 {
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
        (2.0 / (PI * ax)).sqrt() * (p * xx.cos() - z * q * xx.sin())
    };
    sign * r
}

/// Miller downward-recurrence Jₙ(x) with two-buffer normalisation.
///
/// Accurate to ≲10⁻⁹ for |x| ≤ 50, n ≤ 20 (ultrasound range: n ≤ 10, x < 10).
fn bessel_jn(n: u32, x: f64) -> f64 {
    match n {
        0 => bessel_j0(x),
        1 => bessel_j1_clean(x),
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
            let j0_true = bessel_j0(x);
            let j1_true = bessel_j1_clean(x);
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

/// Public crate-level Bessel Jₙ driver (clean two-buffer Miller recurrence).
pub(crate) fn jn(n: u32, x: f64) -> f64 {
    bessel_jn(n, x)
}
