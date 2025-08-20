//! Simple FFT wrappers for Array3

use ndarray::{Array3, Axis};
use num_complex::Complex;
use rustfft::FftPlanner;

/// Perform 3D FFT on a 3D array
pub fn fft_3d_array(field: &Array3<f64>) -> Array3<Complex<f64>> {
    let mut planner = FftPlanner::new();
    let mut result = field.mapv(|v| Complex::new(v, 0.0));
    
    // FFT along each dimension
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

/// Perform 3D IFFT on a 3D complex array
pub fn ifft_3d_array(field_hat: &Array3<Complex<f64>>) -> Array3<f64> {
    let mut planner = FftPlanner::new();
    let mut result = field_hat.clone();
    
    // IFFT along each dimension
    for axis in 0..3 {
        let n = result.shape()[axis];
        let fft = planner.plan_fft_inverse(n);
        
        for mut lane in result.lanes_mut(Axis(axis)) {
            let mut buffer: Vec<Complex<f64>> = lane.iter().cloned().collect();
            fft.process(&mut buffer);
            for (i, val) in buffer.into_iter().enumerate() {
                lane[i] = val / n as f64; // Normalize
            }
        }
    }
    
    result.mapv(|c| c.re)
}