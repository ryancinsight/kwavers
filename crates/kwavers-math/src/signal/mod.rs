//! Signal-processing mathematical primitives.
//!
//! Contains pure mathematical types for signal windowing and apodization,
//! with no dependencies on domain-specific modules.

pub mod phase;
pub mod window;
pub use phase::wrap_to_pi;

use serde::{Deserialize, Serialize};

/// Apodization window type for transducer arrays and beamforming.
///
/// Encodes the window function applied to element amplitude weights to control
/// aperture taper and sidelobe levels. All formula references use symmetric
/// N-point indexing: `i ∈ [0, N−1]`, `center = (N−1)/2`.
///
/// ## Gaussian parameterisation
///
/// `Gaussian { sigma }` uses a normalised half-width `σ ∈ (0, 0.5]`:
///
/// ```text
/// w(i) = exp(−0.5 · ((i − center) / (σ · center))²)
/// ```
///
/// Smaller σ → narrower Gaussian (lower sidelobes, wider main lobe).
/// `σ = 1/3` is a common default (≈ −6 dB at the aperture edge).
///
/// ## Kaiser parameterisation
///
/// `Kaiser { beta }` uses the zeroth-order modified Bessel function I₀:
///
/// ```text
/// w(i) = I₀(β · √(1 − (2i/(N−1) − 1)²)) / I₀(β)
/// ```
///
/// `β = 0` → rectangular; `β ≈ 8.6` → ≈ −80 dB sidelobes.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ApodizationType {
    /// Uniform (rectangular) window — all weights equal to 1.
    Uniform,
    /// Hamming window: w(i) = 0.54 − 0.46·cos(2πi/(N−1))
    Hamming,
    /// Hanning (Hann) window: w(i) = 0.5·(1 − cos(2πi/(N−1)))
    Hanning,
    /// Blackman window: w(i) = 0.42 − 0.5·cos(2πi/(N−1)) + 0.08·cos(4πi/(N−1))
    Blackman,
    /// Tukey (tapered-cosine) window with cosine fraction `r ∈ [0, 1]`:
    /// rectangular at `r = 0`, Hann at `r = 1`. `r` is clamped to `[0, 1]`.
    Tukey { r: f64 },
    /// Kaiser–Bessel window. Higher `beta` trades wider main-lobe for lower
    /// sidelobes; `beta = 8.6` gives ≈ −80 dB.
    Kaiser { beta: f64 },
    /// Gaussian window with normalised half-width `sigma`. Typical values:
    /// `sigma = 0.4` (moderate), `sigma = 1.0/3.0` (narrower), `sigma = 0.5`
    /// (near-rectangular).
    Gaussian { sigma: f64 },
}

impl ApodizationType {
    /// Compute the N-point symmetric apodization weight vector.
    ///
    /// Returns `vec![1.0; 1]` for `n = 0` or `n = 1`.
    #[must_use]
    pub fn weights(&self, n: usize) -> Vec<f64> {
        if n <= 1 {
            return vec![1.0; 1];
        }
        let denom = (n - 1) as f64;
        match *self {
            Self::Uniform => vec![1.0; n],
            // Hann/Hamming/Blackman delegate to the canonical window-coefficient
            // SSOT (`super::window`) so the formulas live in exactly one place.
            Self::Hamming => (0..n).map(|i| window::hamming(i as f64 / denom)).collect(),
            Self::Hanning => (0..n).map(|i| window::hann(i as f64 / denom)).collect(),
            Self::Blackman => (0..n).map(|i| window::blackman(i as f64 / denom)).collect(),
            Self::Tukey { r } => (0..n).map(|i| window::tukey(i as f64 / denom, r)).collect(),
            Self::Kaiser { beta } => {
                let i0_beta = i0(beta);
                (0..n)
                    .map(|i| {
                        let t = 2.0 * i as f64 / denom - 1.0;
                        i0(beta * (1.0 - t * t).max(0.0).sqrt()) / i0_beta
                    })
                    .collect()
            }
            Self::Gaussian { sigma } => {
                let center = denom / 2.0;
                (0..n)
                    .map(|i| {
                        let x = (i as f64 - center) / (sigma * center);
                        (-0.5 * x * x).exp()
                    })
                    .collect()
            }
        }
    }
}

/// Zeroth-order modified Bessel function I₀(x) via polynomial series.
///
/// Converges for all finite x; limited to 30 terms (error < 1e-15 for |x| ≤ 40).
fn i0(x: f64) -> f64 {
    let t = (x / 2.0) * (x / 2.0);
    let mut sum = 1.0_f64;
    let mut term = 1.0_f64;
    for k in 1_u32..=30 {
        term *= t / (k as f64 * k as f64);
        sum += term;
        if term.abs() < 1e-15 * sum.abs() {
            break;
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_all_ones() {
        let w = ApodizationType::Uniform.weights(8);
        assert_eq!(w.len(), 8);
        assert!(w.iter().all(|&x| (x - 1.0).abs() < 1e-15));
    }

    #[test]
    fn hamming_endpoints() {
        let w = ApodizationType::Hamming.weights(64);
        // Hamming endpoints ≈ 0.08 (not zero)
        assert!((w[0] - 0.08).abs() < 1e-10);
        assert!((w[63] - 0.08).abs() < 1e-10);
    }

    #[test]
    fn hanning_zero_endpoints() {
        let w = ApodizationType::Hanning.weights(64);
        assert!(w[0].abs() < 1e-15);
        assert!(w[63].abs() < 1e-15);
    }

    #[test]
    fn blackman_centre_is_one() {
        let n = 65;
        let w = ApodizationType::Blackman.weights(n);
        assert!((w[n / 2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn kaiser_beta_zero_is_rectangular() {
        let w = ApodizationType::Kaiser { beta: 0.0 }.weights(32);
        assert!(w.iter().all(|&x| (x - 1.0).abs() < 1e-10));
    }

    #[test]
    fn gaussian_centre_is_one() {
        let n = 65;
        let w = ApodizationType::Gaussian { sigma: 0.4 }.weights(n);
        assert!((w[n / 2] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn tukey_interpolates_between_rectangular_and_hann() {
        let n = 33;
        // r = 0 ≡ rectangular; r = 1 ≡ Hann.
        let rect = ApodizationType::Uniform.weights(n);
        let t0 = ApodizationType::Tukey { r: 0.0 }.weights(n);
        let hann = ApodizationType::Hanning.weights(n);
        let t1 = ApodizationType::Tukey { r: 1.0 }.weights(n);
        for i in 0..n {
            assert!((t0[i] - rect[i]).abs() < 1e-12, "Tukey(0) == rect at {i}");
            assert!((t1[i] - hann[i]).abs() < 1e-12, "Tukey(1) == Hann at {i}");
        }
        // r = 0.5: zero edges, unity centre, flat interior.
        let t = ApodizationType::Tukey { r: 0.5 }.weights(n);
        assert!(t[0].abs() < 1e-12);
        assert!(t[n - 1].abs() < 1e-12);
        assert!((t[n / 2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn weights_n_one_returns_singleton() {
        for &apod in &[
            ApodizationType::Uniform,
            ApodizationType::Hamming,
            ApodizationType::Hanning,
            ApodizationType::Blackman,
            ApodizationType::Tukey { r: 0.5 },
        ] {
            let w = apod.weights(1);
            assert_eq!(w, vec![1.0]);
        }
    }
}
