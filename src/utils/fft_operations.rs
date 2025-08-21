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
pub fn fft_3d_array(field: &Array3<f64>) -> Array3<Complex<f64>> {
    let mut planner = FftPlanner::new();
    let mut result = field.mapv(|v| Complex::new(v, 0.0));
    
    // FFT along each dimension following standard 3D FFT algorithm
    for axis in 0..3 {
        let n = result.shape()[axis];
        let fft = planner.plan_fft_forward(n);
        
        for mut lane in result.lanes_mut(Axis(axis)) {
            let mut buffer: Vec<Complex<f64>> = lane.iter().cloned().collect();
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
pub fn ifft_3d_array(field_hat: &Array3<Complex<f64>>) -> Array3<f64> {
    let mut planner = FftPlanner::new();
    let mut result = field_hat.clone();
    
    // IFFT along each dimension with proper normalization
    for axis in 0..3 {
        let n = result.shape()[axis];
        let fft = planner.plan_fft_inverse(n);
        
        for mut lane in result.lanes_mut(Axis(axis)) {
            let mut buffer: Vec<Complex<f64>> = lane.iter().cloned().collect();
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
            assert!((val - original[[i, j, k]]).abs() < 1e-10, 
                    "FFT-IFFT identity failed at ({}, {}, {})", i, j, k);
        }
    }
}