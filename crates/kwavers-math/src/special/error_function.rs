//! Gaussian error function — re-exported from the [`leto_ops`] workspace SSOT.

pub use leto_ops::erf;

#[cfg(test)]
mod tests {
    use super::*;
    use eunomia::assert_abs_diff_eq;

    const ERF_AS_TOL: f64 = 1.5e-7;

    #[test]
    fn erf_matches_reference_values() {
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
        assert!(erf(0.5) < erf(1.0) && erf(1.0) < erf(2.0));
    }
}
