//! k-Wave Compatible Utility Functions Module
//!
//! This module provides utility functions compatible with k-Wave toolbox,
//! including angular spectrum propagation, water attenuation models,
//! Hounsfield unit conversions, and other acoustic utilities.
//!
//! # Design Principles
//! - **SOLID**: Single responsibility for each utility function
//! - **DRY**: Reusable implementations across the codebase
//! - **Zero-Copy**: Uses iterators and efficient data structures
//! - **KISS**: Simple, well-documented interfaces
//!
//! # Literature References
//! - Pinkerton (1949): "The absorption of ultrasonic waves in liquids"
//! - Francois & Garrison (1982): "Sound absorption based on ocean measurements"
//! - Goodman (2005): "Introduction to Fourier Optics" (angular spectrum)
//! - Treeby & Cox (2010): "k-Wave: MATLAB toolbox"

use crate::{
    error::{KwaversError, KwaversResult},
    grid::Grid,
};
use ndarray::{Array1, Array2, Array3, Zip, s, Axis};
use rustfft::{FftPlanner, num_complex::Complex};
use std::f64::consts::PI;
use rayon::prelude::*;

/// Angular spectrum propagation method for forward/backward propagation
pub struct AngularSpectrum {
    /// Grid dimensions
    nx: usize,
    ny: usize,
    /// Grid spacing
    dx: f64,
    dy: f64,
    /// Wavenumber arrays
    kx: Array2<f64>,
    ky: Array2<f64>,
    /// FFT planner
    fft_planner: FftPlanner<f64>,
}

impl AngularSpectrum {
    /// Create new angular spectrum propagator
    pub fn new(nx: usize, ny: usize, dx: f64, dy: f64) -> Self {
        // Create wavenumber arrays
        let kx_1d = Self::create_k_vector(nx, dx);
        let ky_1d = Self::create_k_vector(ny, dy);
        
        // Create 2D wavenumber grids
        let mut kx = Array2::zeros((nx, ny));
        let mut ky = Array2::zeros((nx, ny));
        
        for i in 0..nx {
            for j in 0..ny {
                kx[[i, j]] = kx_1d[i];
                ky[[i, j]] = ky_1d[j];
            }
        }
        
        Self {
            nx,
            ny,
            dx,
            dy,
            kx,
            ky,
            fft_planner: FftPlanner::new(),
        }
    }
    
    /// Create k-space vector for FFT
    fn create_k_vector(n: usize, d: f64) -> Vec<f64> {
        let mut k = vec![0.0; n];
        let dk = 2.0 * PI / (n as f64 * d);
        
        for i in 0..n {
            if i <= n / 2 {
                k[i] = i as f64 * dk;
            } else {
                k[i] = (i as f64 - n as f64) * dk;
            }
        }
        
        k
    }
    
    /// Forward propagation using angular spectrum method
    pub fn forward_propagate(
        &mut self,
        field: &Array2<f64>,
        distance: f64,
        wavelength: f64,
    ) -> KwaversResult<Array2<f64>> {
        self.propagate(field, distance, wavelength, true)
    }
    
    /// Backward propagation using angular spectrum method
    pub fn backward_propagate(
        &mut self,
        field: &Array2<f64>,
        distance: f64,
        wavelength: f64,
    ) -> KwaversResult<Array2<f64>> {
        self.propagate(field, distance, wavelength, false)
    }
    
    /// Core propagation function
    fn propagate(
        &mut self,
        field: &Array2<f64>,
        distance: f64,
        wavelength: f64,
        forward: bool,
    ) -> KwaversResult<Array2<f64>> {
        let k = 2.0 * PI / wavelength;
        let sign = if forward { 1.0 } else { -1.0 };
        
        // Convert to complex for FFT
        let mut complex_field: Vec<Complex<f64>> = field.as_slice()
            .unwrap()
            .iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();
        
        // Forward FFT
        let fft = self.fft_planner.plan_fft_forward(self.nx * self.ny);
        fft.process(&mut complex_field);
        
        // Apply propagation in k-space
        for i in 0..self.nx {
            for j in 0..self.ny {
                let idx = i * self.ny + j;
                let kx = self.kx[[i, j]];
                let ky = self.ky[[i, j]];
                let kz_sq = k * k - kx * kx - ky * ky;
                
                if kz_sq > 0.0 {
                    // Propagating waves
                    let kz = kz_sq.sqrt();
                    let phase = Complex::from_polar(1.0, sign * kz * distance);
                    complex_field[idx] *= phase;
                } else {
                    // Evanescent waves - exponential decay
                    let kz_imag = (-kz_sq).sqrt();
                    let decay = (-kz_imag * distance).exp();
                    complex_field[idx] *= decay;
                }
            }
        }
        
        // Inverse FFT
        let ifft = self.fft_planner.plan_fft_inverse(self.nx * self.ny);
        ifft.process(&mut complex_field);
        
        // Extract real part and normalize
        let mut result = Array2::zeros((self.nx, self.ny));
        for i in 0..self.nx {
            for j in 0..self.ny {
                let idx = i * self.ny + j;
                result[[i, j]] = complex_field[idx].re / (self.nx * self.ny) as f64;
            }
        }
        
        Ok(result)
    }
}

/// Water properties and attenuation models
pub struct WaterProperties;

impl WaterProperties {
    /// Calculate water density as function of temperature
    /// Based on Kell (1975) formula
    pub fn density(temperature: f64) -> f64 {
        // Temperature in Celsius
        let t = temperature;
        
        // Kell's formula for water density (kg/m³)
        let a = 999.83952;
        let b = 16.945176;
        let c = -7.9870401e-3;
        let d = -46.170461e-6;
        let e = 105.56302e-9;
        let f = -280.54253e-12;
        let g = 16.879850e-3;
        
        let numerator = a + b * t + c * t.powi(2) + d * t.powi(3) + e * t.powi(4) + f * t.powi(5);
        let denominator = 1.0 + g * t;
        
        numerator / denominator
    }
    
    /// Calculate water sound speed as function of temperature
    /// Based on Bilaniuk & Wong (1993)
    pub fn sound_speed(temperature: f64) -> f64 {
        // Temperature in Celsius
        let t = temperature;
        
        // 5th order polynomial fit
        let c0 = 1402.385;
        let c1 = 5.03830;
        let c2 = -5.81090e-2;
        let c3 = 3.3432e-4;
        let c4 = -1.47797e-6;
        let c5 = 3.1419e-9;
        
        c0 + c1 * t + c2 * t.powi(2) + c3 * t.powi(3) + c4 * t.powi(4) + c5 * t.powi(5)
    }
    
    /// Calculate water absorption coefficient
    /// Based on Francois & Garrison (1982) model
    pub fn absorption_coefficient(
        frequency: f64,      // Hz
        temperature: f64,    // Celsius
        depth: f64,         // meters
        salinity: f64,      // parts per thousand
        ph: f64,            // pH value
    ) -> f64 {
        let f = frequency / 1000.0; // Convert to kHz
        let t = temperature;
        let s = salinity;
        let d = depth / 1000.0; // Convert to km
        
        // Boric acid contribution
        let f1 = 0.78 * (s / 35.0).sqrt() * (t / 26.0).exp();
        let a1 = 8.86 / WaterProperties::sound_speed(t) * 10.0_f64.powf(0.78 * ph - 5.0);
        let p1 = 1.0;
        let boric = a1 * p1 * f1 * f * f / (f1 * f1 + f * f);
        
        // Magnesium sulfate contribution
        let f2 = 42.0 * (t / 17.0).exp();
        let a2 = 21.44 * s / WaterProperties::sound_speed(t) * (1.0 + 0.025 * t);
        let p2 = 1.0 - 1.37e-4 * d + 6.2e-9 * d * d;
        let magnesium = a2 * p2 * f2 * f * f / (f2 * f2 + f * f);
        
        // Pure water contribution
        let a3 = if t <= 20.0 {
            4.937e-4 - 2.59e-5 * t + 9.11e-7 * t * t - 1.50e-8 * t * t * t
        } else {
            3.964e-4 - 1.146e-5 * t + 1.45e-7 * t * t - 6.5e-10 * t * t * t
        };
        let p3 = 1.0 - 3.83e-5 * d + 4.9e-10 * d * d;
        let water = a3 * p3 * f * f;
        
        // Total absorption in dB/km, convert to Np/m
        let alpha_db_per_km = boric + magnesium + water;
        alpha_db_per_km * 0.1151 / 1000.0 // Convert to Np/m
    }
    
    /// Simple Pinkerton model for quick calculations
    pub fn pinkerton_absorption(frequency: f64, temperature: f64) -> f64 {
        // Pinkerton (1949) model: α = A * f²
        // where A depends on temperature
        let f_mhz = frequency / 1e6;
        let a = 25.3 * ((-17.0 / (temperature + 273.15)).exp());
        
        a * f_mhz * f_mhz * 1e-3 // Convert to Np/m
    }
}

/// Hounsfield unit conversions for CT data
pub struct HounsfieldUnits;

impl HounsfieldUnits {
    /// Convert Hounsfield units to density (kg/m³)
    pub fn to_density(hu: f64) -> f64 {
        // Linear relationship: density = 1000 * (1 + HU/1000)
        // Based on water = 0 HU = 1000 kg/m³
        1000.0 * (1.0 + hu / 1000.0)
    }
    
    /// Convert density to Hounsfield units
    pub fn from_density(density: f64) -> f64 {
        // Inverse of to_density
        1000.0 * (density / 1000.0 - 1.0)
    }
    
    /// Convert Hounsfield units to sound speed (m/s)
    /// Based on Mast (2000) empirical relationship
    pub fn to_sound_speed(hu: f64) -> f64 {
        // Empirical relationship for soft tissues
        let density = Self::to_density(hu);
        
        // Mast's formula
        if hu < -100.0 {
            // Fat-like tissue
            1450.0 + 0.5 * hu
        } else if hu < 100.0 {
            // Soft tissue
            1540.0 + 0.3 * hu
        } else {
            // Bone-like tissue
            1580.0 + 1.6 * hu
        }
    }
    
    /// Convert Hounsfield units to acoustic impedance
    pub fn to_impedance(hu: f64) -> f64 {
        let density = Self::to_density(hu);
        let sound_speed = Self::to_sound_speed(hu);
        density * sound_speed
    }
    
    /// Get typical tissue properties from HU value
    pub fn classify_tissue(hu: f64) -> &'static str {
        match hu {
            h if h < -1000.0 => "Air",
            h if h < -100.0 => "Fat",
            h if h < -10.0 => "Water",
            h if h < 40.0 => "Soft Tissue",
            h if h < 100.0 => "Muscle",
            h if h < 300.0 => "Liver",
            h if h < 700.0 => "Trabecular Bone",
            _ => "Cortical Bone",
        }
    }
}

/// Beam pattern calculations
pub struct BeamPatterns;

impl BeamPatterns {
    /// Calculate directivity pattern for circular piston
    pub fn circular_piston_directivity(
        theta: f64,     // Angle from axis (radians)
        radius: f64,    // Piston radius (m)
        wavelength: f64, // Wavelength (m)
    ) -> f64 {
        let k = 2.0 * PI / wavelength;
        let x = k * radius * theta.sin();
        
        if x.abs() < 1e-6 {
            1.0
        } else {
            2.0 * bessel_j1(x) / x
        }
    }
    
    /// Calculate directivity pattern for rectangular piston
    pub fn rectangular_piston_directivity(
        theta_x: f64,   // Angle in x-plane (radians)
        theta_y: f64,   // Angle in y-plane (radians)
        width: f64,     // Width (m)
        height: f64,    // Height (m)
        wavelength: f64, // Wavelength (m)
    ) -> f64 {
        let k = 2.0 * PI / wavelength;
        let x = k * width * theta_x.sin() / 2.0;
        let y = k * height * theta_y.sin() / 2.0;
        
        let dir_x = if x.abs() < 1e-6 { 1.0 } else { x.sin() / x };
        let dir_y = if y.abs() < 1e-6 { 1.0 } else { y.sin() / y };
        
        dir_x * dir_y
    }
    
    /// Calculate beam width at specified level (e.g., -3dB, -6dB)
    pub fn calculate_beam_width(
        transducer_size: f64,
        focal_distance: f64,
        wavelength: f64,
        level_db: f64,
    ) -> f64 {
        // Approximate formula for beam width
        let factor = match level_db {
            l if l >= -3.0 => 0.88,
            l if l >= -6.0 => 1.21,
            l if l >= -20.0 => 2.0,
            _ => 2.5,
        };
        
        factor * wavelength * focal_distance / transducer_size
    }
}

/// Time reversal utilities
pub struct TimeReversalUtils;

impl TimeReversalUtils {
    /// Apply time reversal window
    pub fn apply_time_window(
        signal: &mut Array1<f64>,
        window_type: WindowType,
    ) {
        let n = signal.len();
        
        match window_type {
            WindowType::Tukey(alpha) => {
                for i in 0..n {
                    let x = i as f64 / (n - 1) as f64;
                    let w = if x < alpha / 2.0 {
                        0.5 * (1.0 + (2.0 * PI * x / alpha).cos())
                    } else if x > 1.0 - alpha / 2.0 {
                        0.5 * (1.0 + (2.0 * PI * (1.0 - x) / alpha).cos())
                    } else {
                        1.0
                    };
                    signal[i] *= w;
                }
            }
            WindowType::Exponential(tau) => {
                for i in 0..n {
                    let t = i as f64 / (n - 1) as f64;
                    signal[i] *= (-t / tau).exp();
                }
            }
        }
    }
    
    /// Flip signal for time reversal
    pub fn time_reverse(signal: &Array1<f64>) -> Array1<f64> {
        let mut reversed = signal.clone();
        reversed.invert_axis(Axis(0));
        reversed
    }
    
    /// Apply frequency filter for time reversal
    pub fn frequency_filter(
        signal: &Array1<f64>,
        sampling_freq: f64,
        low_freq: f64,
        high_freq: f64,
    ) -> KwaversResult<Array1<f64>> {
        let n = signal.len();
        let mut fft_planner = FftPlanner::new();
        
        // Convert to complex
        let mut complex_signal: Vec<Complex<f64>> = signal
            .iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();
        
        // FFT
        let fft = fft_planner.plan_fft_forward(n);
        fft.process(&mut complex_signal);
        
        // Apply filter
        let df = sampling_freq / n as f64;
        for i in 0..n {
            let freq = if i <= n / 2 {
                i as f64 * df
            } else {
                (i as f64 - n as f64) * df
            };
            
            if freq.abs() < low_freq || freq.abs() > high_freq {
                complex_signal[i] = Complex::new(0.0, 0.0);
            }
        }
        
        // Inverse FFT
        let ifft = fft_planner.plan_fft_inverse(n);
        ifft.process(&mut complex_signal);
        
        // Extract real part
        Ok(Array1::from_vec(
            complex_signal.iter()
                .map(|c| c.re / n as f64)
                .collect()
        ))
    }
}

/// Window types for signal processing
#[derive(Debug, Clone, Copy)]
pub enum WindowType {
    /// Tukey window with parameter alpha (0 = rectangular, 1 = Hann)
    Tukey(f64),
    /// Exponential decay with time constant tau
    Exponential(f64),
}

/// Numerical utilities
pub struct NumericalUtils;

impl NumericalUtils {
    /// Calculate numerical gradient using central differences
    pub fn gradient_2d(field: &Array2<f64>, dx: f64, dy: f64) -> (Array2<f64>, Array2<f64>) {
        let (nx, ny) = field.dim();
        let mut grad_x = Array2::zeros((nx, ny));
        let mut grad_y = Array2::zeros((nx, ny));
        
        // X-gradient with central differences
        for i in 1..nx-1 {
            for j in 0..ny {
                grad_x[[i, j]] = (field[[i+1, j]] - field[[i-1, j]]) / (2.0 * dx);
            }
        }
        // Forward/backward differences at boundaries
        for j in 0..ny {
            grad_x[[0, j]] = (field[[1, j]] - field[[0, j]]) / dx;
            grad_x[[nx-1, j]] = (field[[nx-1, j]] - field[[nx-2, j]]) / dx;
        }
        
        // Y-gradient with central differences
        for i in 0..nx {
            for j in 1..ny-1 {
                grad_y[[i, j]] = (field[[i, j+1]] - field[[i, j-1]]) / (2.0 * dy);
            }
        }
        // Forward/backward differences at boundaries
        for i in 0..nx {
            grad_y[[i, 0]] = (field[[i, 1]] - field[[i, 0]]) / dy;
            grad_y[[i, ny-1]] = (field[[i, ny-1]] - field[[i, ny-2]]) / dy;
        }
        
        (grad_x, grad_y)
    }
    
    /// Calculate Laplacian using central differences
    pub fn laplacian_2d(field: &Array2<f64>, dx: f64, dy: f64) -> Array2<f64> {
        let (nx, ny) = field.dim();
        let mut laplacian = Array2::zeros((nx, ny));
        
        let dx2 = dx * dx;
        let dy2 = dy * dy;
        
        for i in 1..nx-1 {
            for j in 1..ny-1 {
                laplacian[[i, j]] = (field[[i+1, j]] - 2.0 * field[[i, j]] + field[[i-1, j]]) / dx2 +
                                   (field[[i, j+1]] - 2.0 * field[[i, j]] + field[[i, j-1]]) / dy2;
            }
        }
        
        laplacian
    }
    
    /// Smooth field using Gaussian filter
    pub fn gaussian_smooth_2d(field: &Array2<f64>, sigma: f64) -> Array2<f64> {
        let (nx, ny) = field.dim();
        let kernel_size = (6.0 * sigma).ceil() as usize | 1; // Ensure odd
        let half_size = kernel_size / 2;
        
        // Create Gaussian kernel
        let mut kernel = Array2::zeros((kernel_size, kernel_size));
        let mut sum = 0.0;
        
        for i in 0..kernel_size {
            for j in 0..kernel_size {
                let x = i as f64 - half_size as f64;
                let y = j as f64 - half_size as f64;
                let value = (-(x * x + y * y) / (2.0 * sigma * sigma)).exp();
                kernel[[i, j]] = value;
                sum += value;
            }
        }
        
        // Normalize kernel
        kernel /= sum;
        
        // Apply convolution
        let mut smoothed = Array2::zeros((nx, ny));
        
        for i in 0..nx {
            for j in 0..ny {
                let mut value = 0.0;
                
                for ki in 0..kernel_size {
                    for kj in 0..kernel_size {
                        let ii = (i as i32 + ki as i32 - half_size as i32).max(0).min(nx as i32 - 1) as usize;
                        let jj = (j as i32 + kj as i32 - half_size as i32).max(0).min(ny as i32 - 1) as usize;
                        value += field[[ii, jj]] * kernel[[ki, kj]];
                    }
                }
                
                smoothed[[i, j]] = value;
            }
        }
        
        smoothed
    }
}

/// Bessel function J1 approximation
fn bessel_j1(x: f64) -> f64 {
    // Polynomial approximation for small x
    if x.abs() < 3.0 {
        let x2 = x * x;
        x * (0.5 - x2 / 8.0 + x2 * x2 / 192.0 - x2 * x2 * x2 / 9216.0)
    } else {
        // Asymptotic approximation for large x
        let inv_x = 1.0 / x;
        let phase = x - 3.0 * PI / 4.0;
        (2.0 / (PI * x)).sqrt() * phase.cos() * 
        (1.0 - 3.0 / (8.0 * x * x) + 15.0 / (128.0 * x * x * x * x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_water_properties() {
        // Test water density at 20°C
        let density = WaterProperties::density(20.0);
        assert_relative_eq!(density, 998.2, epsilon = 1.0);
        
        // Test sound speed at 20°C
        let speed = WaterProperties::sound_speed(20.0);
        assert_relative_eq!(speed, 1482.0, epsilon = 5.0);
        
        // Test absorption
        let alpha = WaterProperties::pinkerton_absorption(1e6, 20.0);
        assert!(alpha > 0.0);
    }
    
    #[test]
    fn test_hounsfield_units() {
        // Water should be 0 HU
        let hu_water = HounsfieldUnits::from_density(1000.0);
        assert_relative_eq!(hu_water, 0.0, epsilon = 1e-10);
        
        // Test round-trip conversion
        let hu = 50.0;
        let density = HounsfieldUnits::to_density(hu);
        let hu_back = HounsfieldUnits::from_density(density);
        assert_relative_eq!(hu, hu_back, epsilon = 1e-10);
        
        // Test tissue classification
        assert_eq!(HounsfieldUnits::classify_tissue(-500.0), "Fat");
        assert_eq!(HounsfieldUnits::classify_tissue(30.0), "Soft Tissue");
        assert_eq!(HounsfieldUnits::classify_tissue(1000.0), "Cortical Bone");
    }
    
    #[test]
    fn test_beam_patterns() {
        // Test on-axis directivity
        let dir = BeamPatterns::circular_piston_directivity(0.0, 0.01, 0.001);
        assert_relative_eq!(dir, 1.0, epsilon = 1e-10);
        
        // Test off-axis should be less than on-axis
        let dir_off = BeamPatterns::circular_piston_directivity(0.1, 0.01, 0.001);
        assert!(dir_off < 1.0);
        
        // Test beam width calculation
        let width = BeamPatterns::calculate_beam_width(0.02, 0.1, 0.0015, -3.0);
        assert!(width > 0.0);
    }
    
    #[test]
    fn test_time_reversal() {
        // Test signal reversal
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let reversed = TimeReversalUtils::time_reverse(&signal);
        assert_eq!(reversed[0], 5.0);
        assert_eq!(reversed[4], 1.0);
    }
    
    #[test]
    fn test_numerical_utils() {
        // Test gradient calculation
        let field = Array2::from_shape_fn((5, 5), |(i, j)| i as f64 + j as f64);
        let (grad_x, grad_y) = NumericalUtils::gradient_2d(&field, 1.0, 1.0);
        
        // Interior points should have gradient of 1 in both directions
        assert_relative_eq!(grad_x[[2, 2]], 1.0, epsilon = 1e-10);
        assert_relative_eq!(grad_y[[2, 2]], 1.0, epsilon = 1e-10);
    }
}