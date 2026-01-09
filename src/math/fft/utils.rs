//! FFT and Grid optimization utilities
//!
//! This module provides functions to find optimal grid sizes for FFT performance and other FFT-related utilities.

use ndarray::Array2;
use num_complex::Complex64;

/// Check if a number is a "good" size for FFT (composite of small primes 2, 3, 5)
#[must_use]
pub fn is_optimal_fft_size(n: usize) -> bool {
    if n <= 1 {
        return true;
    }
    let mut num = n;
    while num % 2 == 0 {
        num /= 2;
    }
    while num % 3 == 0 {
        num /= 3;
    }
    while num % 5 == 0 {
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

/// Shift the zero frequency component to the center of the spectrum (FFT shift)
pub fn fft_shift_2d(spectrum: &mut Array2<Complex64>) {
    let (nx, ny) = spectrum.dim();
    let nx_half = nx / 2;
    let ny_half = ny / 2;

    // Create a temporary array for the shifted spectrum
    let mut shifted = Array2::zeros(spectrum.raw_dim());

    // Perform the shift by copying quadrants
    for i in 0..nx_half {
        for j in 0..ny_half {
            // Top-left quadrant -> Bottom-right
            shifted[[i + nx_half, j + ny_half]] = spectrum[[i, j]];
            // Top-right quadrant -> Bottom-left
            shifted[[i + nx_half, j]] = spectrum[[i, j + ny_half]];
            // Bottom-left quadrant -> Top-right
            shifted[[i, j + ny_half]] = spectrum[[i + nx_half, j]];
            // Bottom-right quadrant -> Top-left
            shifted[[i, j]] = spectrum[[i + nx_half, j + ny_half]];
        }
    }

    // Copy back to the original array
    *spectrum = shifted;
}

/// Inverse FFT shift - move zero frequency back to the corner
pub fn ifft_shift_2d(spectrum: &mut Array2<Complex64>) {
    // Inverse shift is the same as forward shift for even-sized arrays
    fft_shift_2d(spectrum);
}
