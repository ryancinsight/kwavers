//! FFT operations for 3D arrays
//!
//! Provides efficient FFT and IFFT operations for acoustic wave simulations.
//! Based on standard DSP literature and validated against known transforms.

use ndarray::{Array3, Axis};
use num_complex::Complex;
use rustfft::FftPlanner;

/// Perform 3D FFT on a real-valued array
///
/// # Arguments
/// * `field` - 3D real-valued array to transform
///
/// # Returns
/// Complex-valued frequency domain representation
///
/// # References
/// - Oppenheim & Schafer, "Discrete-Time Signal Processing", 3rd Ed.
/// - Cooley & Tukey, "An Algorithm for Machine Calculation of Complex Fourier Series", 1965
#[must_use]
pub fn fft_3d_array(field: &Array3<f64>) -> Array3<Complex<f64>> {
    let mut planner = FftPlanner::new();
    let mut result = field.mapv(|v| Complex::new(v, 0.0));

    // FFT along each dimension following standard 3D FFT algorithm
    for axis in 0..3 {
        let n = result.shape()[axis];
        let fft = planner.plan_fft_forward(n);

        for mut lane in result.lanes_mut(Axis(axis)) {
            let mut buffer: Vec<Complex<f64>> = lane.iter().copied().collect();
            fft.process(&mut buffer);
            for (i, val) in buffer.into_iter().enumerate() {
                lane[i] = val;
            }
        }
    }

    result
}

/// Perform 3D IFFT on a complex-valued array
///
/// # Arguments
/// * `field_hat` - 3D complex-valued array in frequency domain
///
/// # Returns
/// Real-valued spatial domain representation
///
/// # Note
/// Applies proper normalization factor of 1/N per dimension
#[must_use]
pub fn ifft_3d_array(field_hat: &Array3<Complex<f64>>) -> Array3<f64> {
    let mut planner = FftPlanner::new();
    let mut result = field_hat.clone();

    // IFFT along each dimension with proper normalization
    for axis in 0..3 {
        let n = result.shape()[axis];
        let fft = planner.plan_fft_inverse(n);

        for mut lane in result.lanes_mut(Axis(axis)) {
            let mut buffer: Vec<Complex<f64>> = lane.iter().copied().collect();
            fft.process(&mut buffer);
            // Apply normalization as per DSP convention
            for (i, val) in buffer.into_iter().enumerate() {
                lane[i] = val / n as f64;
            }
        }
    }

    // Extract real part (imaginary should be negligible for real input)
    result.mapv(|c| c.re)
}

// Legacy compatibility functions for old 4D array interface
// These will be removed in a future refactor when all callers are updated

/// Legacy wrapper for fft_3d that accepts Array4 and extracts a slice
///
/// # Note
/// This is a compatibility shim for legacy code. New code should use fft_3d_array directly.
#[must_use]
#[deprecated(note = "Use fft_3d_array directly with Array3")]
pub fn fft_3d(
    fields: &ndarray::Array4<f64>,
    slice_idx: usize,
    _grid: &crate::grid::Grid,
) -> ndarray::Array3<Complex<f64>> {
    let slice = fields.index_axis(ndarray::Axis(0), slice_idx);
    fft_3d_array(&slice.to_owned())
}

/// Legacy wrapper for ifft_3d that returns a 3D array
///
/// # Note
/// This is a compatibility shim for legacy code. New code should use ifft_3d_array directly.
#[must_use]
#[deprecated(note = "Use ifft_3d_array directly")]
pub fn ifft_3d(
    field_hat: &ndarray::Array3<Complex<f64>>,
    _grid: &crate::grid::Grid,
) -> ndarray::Array3<f64> {
    ifft_3d_array(field_hat)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_fft_ifft_identity() {
        // Parseval's theorem: FFT followed by IFFT should return original
        let original = Array3::<f64>::from_shape_fn((8, 8, 8), |(i, j, k)| {
            (i as f64).sin() * (j as f64).cos() * (k as f64)
        });

        let transformed = fft_3d_array(&original);
        let recovered = ifft_3d_array(&transformed);

        for ((i, j, k), &val) in recovered.indexed_iter() {
            assert!(
                (val - original[[i, j, k]]).abs() < 1e-10,
                "FFT-IFFT identity failed at ({}, {}, {})",
                i,
                j,
                k
            );
        }
    }
}
