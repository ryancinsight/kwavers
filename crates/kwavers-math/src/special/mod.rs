//! Special mathematical functions (workspace SSOT).

pub mod bessel;
mod error_function;
pub mod legendre;

pub use error_function::erf;

/// Unnormalized cardinal sine `sinc(x) = sin(x)/x`, with the removable
/// singularity defined as `sinc(0) = 1` (guarded for `|x| ≤ ε`).
///
/// Note: this is the *unnormalized* convention; for the signal-processing
/// `sin(πx)/(πx)` form, pass `π·x`.
#[inline]
#[must_use]
pub fn sinc(x: f64) -> f64 {
    if x.abs() <= f64::EPSILON {
        1.0
    } else {
        x.sin() / x
    }
}

#[cfg(test)]
mod tests {
    use super::sinc;

    #[test]
    fn sinc_values() {
        assert_eq!(sinc(0.0), 1.0);
        assert!((sinc(std::f64::consts::PI) - 0.0).abs() < 1e-15); // sin(π)/π = 0
        assert!((sinc(1.0) - 1.0_f64.sin()).abs() < 1e-15);
        // Even function.
        assert_eq!(sinc(-0.7), sinc(0.7));
    }
}
