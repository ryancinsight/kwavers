//! Special mathematical functions (workspace SSOT).
//!
//! `sinc`, `erf`, and Bessel functions are re-exported from the
//! [`leto_ops`] SSOT.  Legendre polynomials remain kwavers-specific.

pub mod bessel;
mod error_function;
pub mod legendre;

pub use error_function::erf;
pub use leto_ops::sinc;

#[cfg(test)]
mod tests {
    use super::sinc;

    #[test]
    fn sinc_values() {
        assert_eq!(sinc(0.0), 1.0);
        assert!((sinc(std::f64::consts::PI) - 0.0).abs() < 1e-15);
        assert!((sinc(1.0) - 1.0_f64.sin()).abs() < 1e-15);
        assert_eq!(sinc(-0.7), sinc(0.7));
    }
}
