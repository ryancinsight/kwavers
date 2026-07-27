//! Analytical Bessel J₂ helper for Fubini-absolute literature tests.
//!
//! `J_1` and `J_2` are delegated to the SSOT (`leto_ops::application::special::j1`).

/// Bessel function `J_2(x)` via the recurrence `J_2(x) = (2/x)·J_1(x) − J_0(x)`
/// with `J_0` and `J_1` from their power series.
pub(super) fn bessel_j2(x: f64) -> f64 {
    let j0 = {
        let mut term = 1.0_f64;
        let mut sum = term;
        let half_x_sq = (x * x) / 4.0;
        for k in 1..50 {
            let k_f = k as f64;
            term *= -half_x_sq / (k_f * k_f);
            sum += term;
            if term.abs() < 1.0e-18 {
                break;
            }
        }
        sum
    };
    let j1 = leto_ops::application::special::j1(x);
    if x.abs() < 1.0e-12 {
        return 0.0;
    }
    (2.0 / x) * j1 - j0
}
