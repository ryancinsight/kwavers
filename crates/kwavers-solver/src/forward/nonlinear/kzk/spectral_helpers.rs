//! Shared spectral-index helpers for KZK diffraction operators.
//!
//! All FFT-based diffraction operators need to convert a linear array index
//! into a signed wavenumber that accounts for FFT ordering (positive
//! frequencies first, then negative frequencies in wrap-around order).
//! This module provides a single zero-cast implementation used across every
//! diffraction variant to avoid repeating the same pattern.
//!
//! # Sign convention
//!
//! This module uses the **physics convention**: the Nyquist bin at index `n/2`
//! maps to the **positive** Nyquist frequency `+π/Δx`. This differs from
//! `apollo::fftfreq` (and NumPy's `fftfreq`) which map `n/2` to the negative
//! Nyquist frequency `−π/Δx` for even-length arrays. Use this module
//! (not `Grid::compute_kx/ky/kz`) wherever the positive-Nyquist convention
//! is required by the solver's mathematical derivation.

use kwavers_core::constants::numerical::TWO_PI;

/// Convert FFT-ordered index `i` and array size `n` into the corresponding
/// signed spatial frequency index.
///
/// FFT output convention:
/// - indices `0..=(n/2)` hold zero and positive frequencies
/// - indices `(n/2+1)..(n)` hold negative frequencies in wrap-around order
///
/// Returns the signed integer displacement from zero frequency:
///   `i as isize` for `i <= n/2`
///   `(i as isize - n as isize)` for `i > n/2`
#[inline(always)]
fn signed_spectral_index(i: usize, n: usize) -> isize {
    if i <= n / 2 {
        i as isize
    } else {
        i as isize - n as isize
    }
}

/// Compute the signed wavenumber (rad/m) for FFT-ordered index `i` in a grid
/// of size `n` with physical spacing `d` (m).
///
/// `dk = 2π / (n * d)` and the signed index is computed via
/// [`signed_spectral_index`] to avoid double `as f64` casts in hot loops.
///
/// # Example
///
/// ```
/// # use kwavers_solver::forward::nonlinear::kzk::spectral_helpers::fft_wavenumber;
/// let kx = fft_wavenumber(0, 128, 1e-4);  // zero frequency
/// assert_eq!(kx, 0.0);
/// let kx = fft_wavenumber(64, 128, 1e-4); // Nyquist
/// assert!(kx > 0.0);
/// ```
#[inline(always)]
pub fn fft_wavenumber(i: usize, n: usize, d: f64) -> f64 {
    let dk = TWO_PI / (n as f64 * d);
    signed_spectral_index(i, n) as f64 * dk
}

/// Build a 1-D vector of signed wavenumbers for a grid of size `n` and
/// physical spacing `d`.
///
/// Equivalent to `(0..n).map(|i| fft_wavenumber(i, n, d)).collect()` but
/// avoids recomputing `dk` on every iteration.
#[must_use]
pub fn wavenumber_vec(n: usize, d: f64) -> Vec<f64> {
    let dk = TWO_PI / (n as f64 * d);
    (0..n)
        .map(|i| signed_spectral_index(i, n) as f64 * dk)
        .collect()
}

/// Folded non-negative angular frequency for temporal FFT-ordered index `k`
/// in a grid of `nt` samples with time step `dt`.
///
/// The positive-frequency fold means `ω(nt - k) = ω(k)` for `k > nt/2`.
/// This is the value used by broadband and Padé diffraction operators for the
/// per-harmonic exact propagator.
#[inline(always)]
pub fn folded_omega(k: usize, nt: usize, dt: f64) -> f64 {
    let domega = TWO_PI / (nt as f64 * dt);
    let folded_k = if k <= nt / 2 { k } else { nt - k };
    folded_k as f64 * domega
}
