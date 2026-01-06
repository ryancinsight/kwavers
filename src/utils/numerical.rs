//! Numerical utilities for acoustic simulation
//!
//! Implements various numerical helper functions

use crate::error::KwaversResult;
use crate::signal;
use crate::signal::{window::apply_window as apply_window_slice, window::get_win, WindowType};
use ndarray::Array1;

/// Numerical utility functions
#[derive(Debug)]
pub struct NumericalUtils;

impl NumericalUtils {
    /// Create Gaussian distribution
    #[must_use]
    pub fn gaussian(x: f64, mean: f64, std: f64) -> f64 {
        let exponent = -0.5 * ((x - mean) / std).powi(2);
        (1.0 / (std * (2.0 * std::f64::consts::PI).sqrt())) * exponent.exp()
    }

    /// Create Hann window
    #[must_use]
    pub fn hann_window(n: usize) -> Array1<f64> {
        Array1::from_vec(get_win(WindowType::Hann, n, true))
    }

    /// Apply window function to signal
    pub fn apply_window(signal: &Array1<f64>, window: &Array1<f64>) -> KwaversResult<Array1<f64>> {
        let signal_slice: Vec<f64> = signal.iter().copied().collect();
        let window_slice: Vec<f64> = window.iter().copied().collect();
        Ok(Array1::from_vec(apply_window_slice(
            &signal_slice,
            &window_slice,
        )?))
    }

    /// Calculate next power of 2
    #[must_use]
    pub fn next_pow2(n: usize) -> usize {
        signal::next_pow2(n)
    }

    /// Pad array to specified size
    #[must_use]
    pub fn pad_array(arr: &Array1<f64>, target_size: usize) -> Array1<f64> {
        let slice: Vec<f64> = arr.iter().copied().collect();
        Array1::from_vec(signal::pad_zeros(&slice, target_size))
    }
}
