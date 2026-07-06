//! FFT and spectral-processing utilities.
//!
//! This module keeps non-plan FFT helpers local to `kwavers` while delegating
//! the actual transforms to Apollo.  The helpers here are thin wrappers around
//! the Apollo plan cache and therefore preserve the single source of truth for
//! transform execution while centralizing repeated spectral post-processing.

use crate::fft::{fft_1d_complex_inplace, ifft_1d_complex_inplace};
use ndarray::{Array1, Array2};
use num_complex::Complex64;

/// Check if a number is a "good" size for FFT (composite of small primes 2, 3, 5)
#[must_use]
pub fn is_optimal_fft_size(n: usize) -> bool {
    if n <= 1 {
        return true;
    }
    let mut num = n;
    while num.is_multiple_of(2) {
        num /= 2;
    }
    while num.is_multiple_of(3) {
        num /= 3;
    }
    while num.is_multiple_of(5) {
        num /= 5;
    }
    num == 1
}

/// Find the next optimal FFT size greater than or equal to n
#[must_use]
pub fn get_optimal_fft_size(n: usize) -> usize {
    let mut size = n;
    while !is_optimal_fft_size(size) {
        size += 1;
    }
    size
}

/// Calculate the normalization factor for an inverse FFT
#[must_use]
pub fn ifft_normalization_factor(n: usize) -> f64 {
    if n == 0 {
        0.0
    } else {
        1.0 / n as f64
    }
}

/// Shift the zero frequency component to the center of the spectrum (`fftshift`).
pub fn fft_shift_2d(spectrum: &mut Array2<Complex64>) {
    let (nx, ny) = spectrum.dim();
    let shift_x = nx.div_ceil(2);
    let shift_y = ny.div_ceil(2);
    let mut shifted = Array2::zeros(spectrum.raw_dim());
    for i in 0..nx {
        let src_i = (i + shift_x) % nx;
        for j in 0..ny {
            let src_j = (j + shift_y) % ny;
            shifted[[i, j]] = spectrum[[src_i, src_j]];
        }
    }
    *spectrum = shifted;
}

/// Inverse FFT shift - move zero frequency back to the corner (`ifftshift`).
pub fn ifft_shift_2d(spectrum: &mut Array2<Complex64>) {
    let (nx, ny) = spectrum.dim();
    let shift_x = nx / 2;
    let shift_y = ny / 2;
    let mut shifted = Array2::zeros(spectrum.raw_dim());
    for i in 0..nx {
        let src_i = (i + shift_x) % nx;
        for j in 0..ny {
            let src_j = (j + shift_y) % ny;
            shifted[[i, j]] = spectrum[[src_i, src_j]];
        }
    }
    *spectrum = shifted;
}

/// Apply a real-valued frequency response to a 1-D trace using the Apollo
/// FFT cache.
///
/// ## Theorem
/// Let `X[k]` be the discrete Fourier transform of `x[n]`.  For any real
/// frequency response `H[k]`, the inverse transform of `H[k]X[k]` is the
/// discrete linear filter induced by the convolution theorem.
///
/// ## Proof sketch
/// The Apollo plan computes the unnormalized DFT pair.  Multiplication in the
/// spectral domain and inverse transformation therefore correspond exactly to
/// circular convolution in the discrete domain.  The caller supplies the
/// desired frequency response, so this helper is a thin implementation of the
/// theorem rather than a heuristic approximation.
#[must_use]
pub fn apply_spectral_response_1d<F>(
    signal: &Array1<f64>,
    sampling_frequency: f64,
    mut response: F,
) -> Array1<f64>
where
    F: FnMut(usize, f64, f64) -> f64,
{
    let n = signal.len();
    if n == 0 {
        return Array1::zeros(0);
    }

    let mut spectrum = signal.mapv(|value| Complex64::new(value, 0.0));
    fft_1d_complex_inplace(&mut spectrum);

    let df = sampling_frequency / n as f64;
    let nyquist = sampling_frequency / 2.0;
    for (idx, coeff) in spectrum.iter_mut().enumerate() {
        let freq = idx as f64 * df;
        *coeff *= response(idx, freq, nyquist);
    }

    // Apollo uses FFTW-compatible 1/N inverse normalisation;
    // no additional scaling is required.
    ifft_1d_complex_inplace(&mut spectrum);
    Array1::from_shape_fn(n, |idx| spectrum[idx].re)
}

/// Compute the discrete analytic signal of a real trace.
///
/// ## Theorem
/// For a real sequence `x[n]`, the mask `[1, 2, ..., 2, 1, 0, ..., 0]` applied
/// to its discrete spectrum yields the analytic signal whose real part equals
/// `x[n]` and whose imaginary part is the discrete Hilbert transform.
///
/// ## Proof sketch
/// The discrete Hilbert transform suppresses the negative-frequency half of
/// the spectrum and doubles the positive-frequency half.  The inverse DFT then
/// reconstructs the complex analytic signal.  DC and Nyquist bins remain
/// unchanged to preserve the real-valued symmetry constraints for even-length
/// sequences.
#[must_use]
pub fn analytic_signal_1d(signal: &Array1<f64>) -> Array1<Complex64> {
    let n = signal.len();
    if n == 0 {
        return Array1::zeros(0);
    }

    let mut spectrum = signal.mapv(|value| Complex64::new(value, 0.0));
    fft_1d_complex_inplace(&mut spectrum);

    if n > 1 {
        let half = n / 2;
        if n.is_multiple_of(2) {
            for coeff in spectrum.iter_mut().take(half).skip(1) {
                *coeff *= 2.0;
            }
            for coeff in spectrum.iter_mut().skip(half + 1) {
                *coeff = Complex64::default();
            }
        } else {
            for coeff in spectrum.iter_mut().take(half + 1).skip(1) {
                *coeff *= 2.0;
            }
            for coeff in spectrum.iter_mut().skip(half + 1) {
                *coeff = Complex64::default();
            }
        }
    }

    // Apollo uses FFTW-compatible 1/N inverse normalisation;
    // no additional scaling is required.
    ifft_1d_complex_inplace(&mut spectrum);
    spectrum
}

#[cfg(test)]
mod tests {
    use super::{analytic_signal_1d, apply_spectral_response_1d, fft_shift_2d, ifft_shift_2d};
    use ndarray::{Array1, Array2};
    use num_complex::Complex64;

    #[test]
    fn test_fft_shift_2d_matches_numpy_convention_for_odd_sizes() {
        let mut spectrum =
            Array2::from_shape_fn((3, 5), |(i, j)| Complex64::new((10 * i + j) as f64, 0.0));
        let original = spectrum.clone();
        fft_shift_2d(&mut spectrum);

        let expected = Array2::from_shape_fn((3, 5), |(i, j)| {
            let src_i = (i + 3_usize.div_ceil(2)) % 3;
            let src_j = (j + 5_usize.div_ceil(2)) % 5;
            original[[src_i, src_j]]
        });
        assert_eq!(spectrum, expected);
    }

    #[test]
    fn test_ifft_shift_2d_is_inverse_of_fft_shift_2d_for_odd_sizes() {
        let mut spectrum =
            Array2::from_shape_fn((5, 3), |(i, j)| Complex64::new((10 * i + j) as f64, 0.0));
        let original = spectrum.clone();
        fft_shift_2d(&mut spectrum);
        ifft_shift_2d(&mut spectrum);
        assert_eq!(spectrum, original);
    }

    #[test]
    fn test_apply_spectral_response_1d_preserves_constant_gain() {
        let signal = Array1::from_vec(vec![1.0, 0.0, -1.0, 0.0]);
        let filtered = apply_spectral_response_1d(&signal, 4.0, |_, _, _| 1.0);
        assert_eq!(filtered.len(), signal.len());
        for (lhs, rhs) in filtered.iter().zip(signal.iter()) {
            assert!((lhs - rhs).abs() < 1e-12);
        }
    }

    #[test]
    fn test_analytic_signal_1d_has_unit_envelope_for_bin_centered_tone() {
        use std::f64::consts::TAU;

        let n = 16;
        let signal = Array1::from_shape_fn(n, |k| (TAU * 2.0 * k as f64 / n as f64).cos());
        let analytic = analytic_signal_1d(&signal);

        for (value, &source) in analytic.iter().zip(signal.iter()) {
            let envelope = value.norm();
            assert!(
                (envelope - 1.0).abs() < 1e-10,
                "envelope mismatch for {source}"
            );
        }
    }
}
