//! Bessel functions of the first kind, Jₙ(x) — re-exported from the
//! [`leto_ops`] workspace SSOT.
//!
//! References: Abramowitz & Stegun (1964) §9; Press et al., *Numerical Recipes*;
//! DLMF 10.

pub use leto_ops::application::special::{j0, j1};

/// Bessel Jₙ(x) for integer order `n ≥ 0`.
///
/// Thin wrapper around [`leto_ops::jn`] that accepts `u32` order (the kwavers
/// convention) and delegates to the `usize`-typed SSOT.
#[must_use]
pub fn jn(n: u32, x: f64) -> f64 {
    leto_ops::jn(n as usize, x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_values() {
        assert_eq!(j0(0.0), 1.0);
        assert_eq!(j1(0.0), 0.0);
        assert_eq!(jn(0, 0.0), 1.0);
        assert_eq!(jn(5, 0.0), 0.0);
        assert!((j0(1.0) - 0.765_197_686_6).abs() < 1e-8);
        assert!((j1(1.0) - 0.440_050_585_7).abs() < 1e-8);
        assert!((j0(5.0) - (-0.177_596_771_3)).abs() < 1e-8);
        assert!((j1(5.0) - (-0.327_579_137_9)).abs() < 1e-8);
        assert!((jn(2, 1.0) - 0.114_903_484_9).abs() < 1e-8);
        assert!((jn(3, 2.0) - 0.128_943_249_8).abs() < 1e-8);
    }

    #[test]
    fn parity() {
        for &x in &[0.3, 1.7, 4.2, 9.5] {
            assert!((j0(-x) - j0(x)).abs() < 1e-14);
            assert!((j1(-x) + j1(x)).abs() < 1e-14);
        }
    }
}
