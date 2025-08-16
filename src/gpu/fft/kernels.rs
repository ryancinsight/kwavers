//! Shared FFT kernel algorithms
//!
//! Contains platform-independent FFT algorithms and utilities

use num_complex::Complex;
use std::f64::consts::PI;

/// FFT kernel trait for different implementations
pub trait FftKernel {
    /// Perform 1D FFT on a slice of complex numbers
    fn fft_1d(&mut self, data: &mut [Complex<f64>], forward: bool);
    
    /// Compute twiddle factors for FFT
    fn compute_twiddle_factors(&self, n: usize) -> Vec<Complex<f64>>;
}

/// Cooley-Tukey FFT implementation
pub struct CooleyTukeyFft;

impl FftKernel for CooleyTukeyFft {
    fn fft_1d(&mut self, data: &mut [Complex<f64>], forward: bool) {
        let n = data.len();
        if n <= 1 {
            return;
        }
        
        // Bit-reversal permutation
        bit_reverse_permutation(data);
        
        // Cooley-Tukey FFT
        let mut size = 2;
        while size <= n {
            let half_size = size / 2;
            let angle_sign = if forward { -1.0 } else { 1.0 };
            let angle = angle_sign * 2.0 * PI / size as f64;
            let w = Complex::new(angle.cos(), angle.sin());
            
            for start in (0..n).step_by(size) {
                let mut w_n = Complex::new(1.0, 0.0);
                for k in 0..half_size {
                    let t = data[start + k + half_size] * w_n;
                    data[start + k + half_size] = data[start + k] - t;
                    data[start + k] = data[start + k] + t;
                    w_n = w_n * w;
                }
            }
            size *= 2;
        }
        
        // Normalize for inverse FFT
        if !forward {
            let norm = 1.0 / n as f64;
            for x in data.iter_mut() {
                *x = *x * norm;
            }
        }
    }
    
    fn compute_twiddle_factors(&self, n: usize) -> Vec<Complex<f64>> {
        (0..n)
            .map(|k| {
                let angle = -2.0 * PI * k as f64 / n as f64;
                Complex::new(angle.cos(), angle.sin())
            })
            .collect()
    }
}

/// Perform bit-reversal permutation on data
fn bit_reverse_permutation(data: &mut [Complex<f64>]) {
    let n = data.len();
    let mut j = 0;
    
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        
        if i < j {
            data.swap(i, j);
        }
    }
}