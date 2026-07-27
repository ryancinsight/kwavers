//! Special mathematical functions (workspace SSOT).
//!
//! All functions are re-exported from the [`leto_ops`] SSOT:
//! `sinc`, `erf`, `j0`, `j1`, `jn`, `legendre_poly`, `legendre_poly_and_deriv`.

pub use leto_ops::application::special::{erf, j0, j1, jn, sinc};
pub use leto_ops::application::special_legendre::{legendre_poly, legendre_poly_and_deriv};

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
