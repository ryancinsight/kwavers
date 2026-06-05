//! Canonical window-coefficient functions (workspace SSOT).
//!
//! Each takes a **normalized symmetric position** `x = i/(N−1) ∈ [0, 1]` and
//! returns the window weight. Hosting these here (the foundation `kwavers-math`
//! layer) lets both [`super::ApodizationType`] and the domain-level
//! `kwavers_signal::window` evaluate identical formulas without duplication.
//!
//! References: Harris (1978), "On the Use of Windows for Harmonic Analysis with
//! the Discrete Fourier Transform", Proc. IEEE 66(1).

use kwavers_core::constants::numerical::{FOUR_PI, TWO_PI};

/// Hann window: `w(x) = 0.5·(1 − cos(2πx))`.
#[inline]
#[must_use]
pub fn hann(x: f64) -> f64 {
    0.5 * (1.0 - (TWO_PI * x).cos())
}

/// Hamming window: `w(x) = 0.54 − 0.46·cos(2πx)`.
#[inline]
#[must_use]
pub fn hamming(x: f64) -> f64 {
    0.46f64.mul_add(-(TWO_PI * x).cos(), 0.54)
}

/// Blackman window: `w(x) = 0.42 − 0.5·cos(2πx) + 0.08·cos(4πx)`.
#[inline]
#[must_use]
pub fn blackman(x: f64) -> f64 {
    0.08f64.mul_add(
        (FOUR_PI * x).cos(),
        0.5f64.mul_add(-(TWO_PI * x).cos(), 0.42),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoints_and_center() {
        // Hann is 0 at the symmetric endpoints and 1 at the center.
        assert!((hann(0.0)).abs() < 1e-12);
        assert!((hann(1.0)).abs() < 1e-12);
        assert!((hann(0.5) - 1.0).abs() < 1e-12);
        // Hamming pedestal: 0.08 at the endpoints.
        assert!((hamming(0.0) - 0.08).abs() < 1e-12);
        assert!((hamming(0.5) - 1.0).abs() < 1e-12);
        // Blackman is ~0 at the endpoints (0.42 − 0.5 + 0.08 = 0).
        assert!((blackman(0.0)).abs() < 1e-12);
        assert!((blackman(0.5) - 1.0).abs() < 1e-12);
    }
}
