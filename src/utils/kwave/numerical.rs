//! Numerical utilities for k-Wave compatibility
//!
//! Implements various numerical helper functions

use crate::error::KwaversResult;
use ndarray::{Array1, Array2, Array3, Axis};

/// Numerical utility functions
#[derive(Debug)]
pub struct NumericalUtils;

impl NumericalUtils {
    /// Create Gaussian distribution
    pub fn gaussian(x: f64, mean: f64, std: f64) -> f64 {
        let exponent = -0.5 * ((x - mean) / std).powi(2);
        (1.0 / (std * (2.0 * std::f64::consts::PI).sqrt())) * exponent.exp()
    }

    /// Create Hann window
    pub fn hann_window(n: usize) -> Array1<f64> {
        use std::f64::consts::PI;
        Array1::from_shape_fn(n, |i| {
            0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos())
        })
    }

    /// Apply window function to signal
    pub fn apply_window(signal: &Array1<f64>, window: &Array1<f64>) -> KwaversResult<Array1<f64>> {
        if signal.len() != window.len() {
            return Err(crate::error::KwaversError::Numerical(
                crate::error::NumericalError::MatrixDimension {
                    operation: "apply_window".to_string(),
                    expected: format!("length {}", signal.len()),
                    actual: format!("length {}", window.len()),
                }
            ));
        }
        Ok(signal * window)
    }

    /// Calculate next power of 2
    pub fn next_pow2(n: usize) -> usize {
        let mut p = 1;
        while p < n {
            p <<= 1;
        }
        p
    }

    /// Pad array to specified size
    pub fn pad_array(arr: &Array1<f64>, target_size: usize) -> Array1<f64> {
        let mut padded = Array1::zeros(target_size);
        let copy_size = arr.len().min(target_size);
        padded.slice_mut(ndarray::s![..copy_size]).assign(&arr.slice(ndarray::s![..copy_size]));
        padded
    }
}