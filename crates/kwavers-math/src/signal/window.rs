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

/// Tukey (tapered-cosine) window with cosine fraction `r ∈ [0, 1]`: cosine
/// tapers over the outer `r` fraction of the span, flat (`= 1`) in between.
///
/// `r` is clamped to `[0, 1]`. `r = 0` degenerates to the rectangular window
/// (`w ≡ 1`); `r = 1` recovers the Hann window. For `x = i/(N−1) ∈ [0, 1]`:
/// ```text
/// w(x) = 0.5·(1 + cos((2π/r)·(x − r/2)))         for x < r/2
///      = 1                                        for r/2 ≤ x ≤ 1 − r/2
///      = 0.5·(1 + cos((2π/r)·(x − 1 + r/2)))      for x > 1 − r/2
/// ```
#[inline]
#[must_use]
pub fn tukey(x: f64, r: f64) -> f64 {
    let r = r.clamp(0.0, 1.0);
    if r == 0.0 {
        return 1.0;
    }
    let half = 0.5 * r;
    if x < half {
        0.5 * (1.0 + (TWO_PI / r * (x - half)).cos())
    } else if x <= 1.0 - half {
        1.0
    } else {
        0.5 * (1.0 + (TWO_PI / r * (x - 1.0 + half)).cos())
    }
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

    #[test]
    fn tukey_degenerates_to_rectangular_and_hann() {
        // r = 0 → rectangular (≡ 1) everywhere.
        for &x in &[0.0, 0.25, 0.5, 0.75, 1.0] {
            assert!((tukey(x, 0.0) - 1.0).abs() < 1e-12, "r=0 at x={x}");
        }
        // r = 1 → Hann at every position.
        for k in 0..=10 {
            let x = k as f64 / 10.0;
            assert!((tukey(x, 1.0) - hann(x)).abs() < 1e-12, "r=1 vs hann at x={x}");
        }
        // r is clamped: r > 1 behaves like r = 1.
        assert!((tukey(0.3, 2.0) - tukey(0.3, 1.0)).abs() < 1e-12);
    }

    #[test]
    fn tukey_taper_and_flat_top() {
        let r = 0.5; // half = 0.25
        // Zero at the symmetric endpoints, unity over the flat interior.
        assert!((tukey(0.0, r)).abs() < 1e-12);
        assert!((tukey(1.0, r)).abs() < 1e-12);
        assert!((tukey(0.25, r) - 1.0).abs() < 1e-12); // taper meets flat top
        assert!((tukey(0.5, r) - 1.0).abs() < 1e-12); // center is flat
        assert!((tukey(0.75, r) - 1.0).abs() < 1e-12); // flat top meets taper
        // Symmetric about x = 0.5.
        assert!((tukey(0.1, r) - tukey(0.9, r)).abs() < 1e-12);
    }
}
