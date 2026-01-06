//! FFT and Grid optimization utilities
//!
//! This module provides functions to find optimal grid sizes for FFT performance and other FFT-related utilities.

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
