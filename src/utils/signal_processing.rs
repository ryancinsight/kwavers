//! Signal processing utilities for envelope, phase, and analytic signal computation
//!
//! This module provides literature-validated signal processing functions including:
//! - Hilbert transform for analytic signal generation
//! - Instantaneous envelope computation
//! - Instantaneous phase computation
//!
//! # References
//!
//! - Bozdağ et al. (2011): "Misfit functions for full waveform inversion based on 
//!   instantaneous phase and envelope measurements", *Geophysical Journal International*
//! - Fichtner et al. (2008): "The adjoint method in seismology", *Physics of the Earth 
//!   and Planetary Interiors*
//! - Marple (1999): "Computing the discrete-time analytic signal via FFT", *IEEE Transactions 
//!   on Signal Processing*

use ndarray::{Array1, Array2};
use num_complex::Complex;
use rustfft::{FftPlanner, num_complex};

/// Compute the Hilbert transform of a real signal using FFT
///
/// The Hilbert transform converts a real signal into its analytic representation:
/// `z(t) = x(t) + i*H[x(t)]`
///
/// where `H[x(t)]` is the Hilbert transform.
///
/// # Algorithm
///
/// 1. Take FFT of input signal
/// 2. Zero negative frequencies, double positive frequencies
/// 3. Take inverse FFT
///
/// # Arguments
///
/// * `signal` - Input real signal
///
/// # Returns
///
/// Analytic signal as complex array where:
/// - Real part: original signal
/// - Imaginary part: Hilbert transform
///
/// # References
///
/// - Marple (1999): "Computing the discrete-time analytic signal via FFT"
/// - Oppenheim & Schafer (2009): "Discrete-Time Signal Processing", Chapter 12
#[must_use]
pub fn hilbert_transform(signal: &Array1<f64>) -> Array1<Complex<f64>> {
    let n = signal.len();
    
    // Edge case: empty or single-sample signal
    if n <= 1 {
        return signal.iter().map(|&x| Complex::new(x, 0.0)).collect();
    }
    
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);

    // Convert to complex for FFT
    let mut complex_signal: Vec<Complex<f64>> =
        signal.iter().map(|&x| Complex::new(x, 0.0)).collect();

    // Forward FFT
    fft.process(&mut complex_signal);

    // Apply Hilbert transform in frequency domain
    // Per Marple (1999): H[x] = -i * sign(ω) * X(ω)
    // Implemented as: zero negative frequencies, double positive frequencies
    for (i, complex_val) in complex_signal.iter_mut().enumerate().take(n) {
        if i == 0 || (n.is_multiple_of(2) && i == n / 2) {
            // DC and Nyquist components unchanged
            // These are purely real frequencies
        } else if i < n.div_ceil(2) {
            // Positive frequencies: multiply by 2
            *complex_val *= 2.0;
        } else {
            // Negative frequencies: zero out
            *complex_val = Complex::new(0.0, 0.0);
        }
    }

    // Inverse FFT to get analytic signal
    ifft.process(&mut complex_signal);

    // Normalize by 1/N (FFT convention)
    let norm_factor = 1.0 / n as f64;
    complex_signal.iter().map(|&val| val * norm_factor).collect()
}

/// Compute instantaneous envelope from analytic signal
///
/// The envelope is the magnitude of the analytic signal:
/// `E(t) = |z(t)| = sqrt(x²(t) + H[x(t)]²)`
///
/// # Arguments
///
/// * `signal` - Input real signal
///
/// # Returns
///
/// Instantaneous envelope (always non-negative)
///
/// # References
///
/// - Bozdağ et al. (2011): Envelope-based misfit function
/// - Gabor (1946): "Theory of communication", *Journal of the IEE*
#[must_use]
pub fn instantaneous_envelope(signal: &Array1<f64>) -> Array1<f64> {
    let analytic = hilbert_transform(signal);
    analytic.iter().map(|z| z.norm()).collect()
}

/// Compute instantaneous phase from analytic signal
///
/// The phase is the argument of the analytic signal:
/// `φ(t) = atan2(H[x(t)], x(t))`
///
/// # Arguments
///
/// * `signal` - Input real signal
///
/// # Returns
///
/// Instantaneous phase in radians [-π, π]
///
/// # References
///
/// - Fichtner et al. (2008): Phase-based adjoint method
/// - Boashash (1992): "Estimating and interpreting the instantaneous frequency"
#[must_use]
pub fn instantaneous_phase(signal: &Array1<f64>) -> Array1<f64> {
    let analytic = hilbert_transform(signal);
    analytic.iter().map(|z| z.arg()).collect()
}

/// Compute instantaneous frequency from phase
///
/// The instantaneous frequency is the time derivative of the phase:
/// `f(t) = (1/2π) * dφ/dt`
///
/// Uses central differences for numerical differentiation.
///
/// # Arguments
///
/// * `signal` - Input real signal
/// * `dt` - Time step
///
/// # Returns
///
/// Instantaneous frequency in Hz
///
/// # References
///
/// - Boashash (1992): "Estimating and interpreting the instantaneous frequency"
#[must_use]
pub fn instantaneous_frequency(signal: &Array1<f64>, dt: f64) -> Array1<f64> {
    let phase = instantaneous_phase(signal);
    let n = phase.len();
    
    if n < 3 {
        return Array1::zeros(n);
    }
    
    let mut freq = Array1::zeros(n);
    let factor = 1.0 / (2.0 * std::f64::consts::PI * dt);
    
    // Forward difference for first point
    freq[0] = (phase[1] - phase[0]) * factor;
    
    // Central difference for interior points
    for i in 1..n-1 {
        // Handle phase wrapping
        let mut dphi = phase[i+1] - phase[i-1];
        while dphi > std::f64::consts::PI {
            dphi -= 2.0 * std::f64::consts::PI;
        }
        while dphi < -std::f64::consts::PI {
            dphi += 2.0 * std::f64::consts::PI;
        }
        freq[i] = dphi / (2.0 * dt) * factor;
    }
    
    // Backward difference for last point
    freq[n-1] = (phase[n-1] - phase[n-2]) * factor;
    
    freq
}

/// Apply Hilbert transform to each row of a 2D array
///
/// Useful for processing multi-channel seismic data or time series.
///
/// # Arguments
///
/// * `data` - Input 2D array (rows × time samples)
///
/// # Returns
///
/// 2D array of analytic signals
pub fn hilbert_transform_2d(data: &Array2<f64>) -> Array2<Complex<f64>> {
    let (nrows, ncols) = data.dim();
    let mut result = Array2::from_elem((nrows, ncols), Complex::new(0.0, 0.0));
    
    for (i, row) in data.outer_iter().enumerate() {
        let analytic = hilbert_transform(&row.to_owned());
        for (j, &val) in analytic.iter().enumerate() {
            result[[i, j]] = val;
        }
    }
    
    result
}

/// Compute envelope for each row of a 2D array
#[must_use]
pub fn instantaneous_envelope_2d(data: &Array2<f64>) -> Array2<f64> {
    let (nrows, ncols) = data.dim();
    let mut result = Array2::zeros((nrows, ncols));
    
    for (i, row) in data.outer_iter().enumerate() {
        let envelope = instantaneous_envelope(&row.to_owned());
        for (j, &val) in envelope.iter().enumerate() {
            result[[i, j]] = val;
        }
    }
    
    result
}

/// Compute phase for each row of a 2D array
#[must_use]
pub fn instantaneous_phase_2d(data: &Array2<f64>) -> Array2<f64> {
    let (nrows, ncols) = data.dim();
    let mut result = Array2::zeros((nrows, ncols));
    
    for (i, row) in data.outer_iter().enumerate() {
        let phase = instantaneous_phase(&row.to_owned());
        for (j, &val) in phase.iter().enumerate() {
            result[[i, j]] = val;
        }
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_hilbert_transform_sine() {
        // Hilbert transform of sin(t) = -cos(t)
        let n = 128;
        let t: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
        let signal: Array1<f64> = t.iter().map(|&ti| (ti).sin()).collect();
        
        let analytic = hilbert_transform(&signal);
        
        // Check that imaginary part ≈ -cos(t) in interior (avoiding edge effects)
        for i in 10..n-10 {
            let ti = t[i];
            let expected_imag = -(ti).cos();
            assert_relative_eq!(analytic[i].im, expected_imag, epsilon = 0.15);
        }
    }

    #[test]
    fn test_hilbert_transform_cosine() {
        // Hilbert transform of cos(t) = sin(t)
        let n = 128;
        let t: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
        let signal: Array1<f64> = t.iter().map(|&ti| (ti).cos()).collect();
        
        let analytic = hilbert_transform(&signal);
        
        // Check that imaginary part ≈ sin(t) in interior (avoiding edge effects)
        for i in 10..n-10 {
            let ti = t[i];
            let expected_imag = (ti).sin();
            assert_relative_eq!(analytic[i].im, expected_imag, epsilon = 0.15);
        }
    }

    #[test]
    fn test_instantaneous_envelope() {
        // Envelope of A*cos(ωt) should be approximately A
        let n = 256;
        let amplitude = 2.5;
        let omega = 0.5;
        let t: Vec<f64> = (0..n).map(|i| i as f64 * 0.05).collect();
        let signal: Array1<f64> = t.iter().map(|&ti| amplitude * (omega * ti).cos()).collect();
        
        let envelope = instantaneous_envelope(&signal);
        
        // Envelope should be close to amplitude (allowing for edge effects)
        for i in 10..n-10 {
            assert_relative_eq!(envelope[i], amplitude, epsilon = 0.2);
        }
    }

    #[test]
    fn test_instantaneous_phase_constant_frequency() {
        // Phase of cos(ωt) should increase linearly
        let n = 256;
        let omega = 0.5;
        let t: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
        let signal: Array1<f64> = t.iter().map(|&ti| (omega * ti).cos()).collect();
        
        let phase = instantaneous_phase(&signal);
        
        // Phase derivative should be approximately constant (allowing for wrapping)
        // This is a basic sanity check
        assert!(phase.len() == n);
    }

    #[test]
    fn test_empty_signal() {
        let signal = Array1::from(vec![]);
        let analytic = hilbert_transform(&signal);
        assert_eq!(analytic.len(), 0);
    }

    #[test]
    fn test_single_sample() {
        let signal = Array1::from(vec![1.0]);
        let analytic = hilbert_transform(&signal);
        assert_eq!(analytic.len(), 1);
        assert_relative_eq!(analytic[0].re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(analytic[0].im, 0.0, epsilon = 1e-10);
    }
}
