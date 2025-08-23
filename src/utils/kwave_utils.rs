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
//! - **KISS**: Clear, well-documented interfaces
//!
//! # Literature References
//! - Pinkerton (1949): "The absorption of ultrasonic waves in liquids"
//! - Francois & Garrison (1982): "Sound absorption based on ocean measurements"
//! - Goodman (2005): "Introduction to Fourier Optics" (angular spectrum)
//! - Treeby & Cox (2010): "k-Wave: MATLAB toolbox"

use crate::error::KwaversResult;
use ndarray::{Array1, Array2, Axis};
use rustfft::{num_complex::Complex, FftPlanner};
use std::f64::consts::PI;

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
        let mut complex_field: Vec<Complex<f64>> = field
            .as_slice()
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
    // Sound speed polynomial coefficients (Bilaniuk & Wong, 1993)
    const SOUND_SPEED_C0: f64 = 1402.385;
    const SOUND_SPEED_C1: f64 = 5.03830;
    const SOUND_SPEED_C2: f64 = -5.81090e-2;
    const SOUND_SPEED_C3: f64 = 3.34320e-4;
    const SOUND_SPEED_C4: f64 = -1.48259e-6;
    const SOUND_SPEED_C5: f64 = 3.16090e-9;

    // ------------------------------------------------------------------------
    // Water density polynomial coefficients (Kell, 1975)
    // Reference: Kell, G. S. (1975). "Density, thermal expansivity, and compressibility of liquid water from 0° to 150°C: Correlations and tables for atmospheric pressure and saturation reviewed and expressed on 1968 temperature scale." J. Chem. Eng. Data, 20(1), 97–105.
    // Units: All coefficients in kg/m³, temperature in °C
    /// Constant term [kg/m³]
    const KELL_A: f64 = 999.83952;
    /// Linear coefficient [kg/m³·°C⁻¹]
    const KELL_B: f64 = 16.945176;
    /// Quadratic coefficient [kg/m³·°C⁻²]
    const KELL_C: f64 = -7.9870401e-3;
    /// Cubic coefficient [kg/m³·°C⁻³]
    const KELL_D: f64 = -46.170461e-6;
    /// Quartic coefficient [kg/m³·°C⁻⁴]
    const KELL_E: f64 = 105.56302e-9;
    /// Quintic coefficient [kg/m³·°C⁻⁵]
    const KELL_F: f64 = -280.54253e-12;
    /// Denominator linear coefficient [dimensionless]
    const KELL_G: f64 = 16.879850e-3;

    /// Calculate water density as function of temperature
    /// Based on Kell (1975) formula
    pub fn density(temperature: f64) -> f64 {
        // Temperature in Celsius
        let t = temperature;

        // Kell's formula for water density (kg/m³)
        let numerator = Self::KELL_A
            + Self::KELL_B * t
            + Self::KELL_C * t.powi(2)
            + Self::KELL_D * t.powi(3)
            + Self::KELL_E * t.powi(4)
            + Self::KELL_F * t.powi(5);
        let denominator = 1.0 + Self::KELL_G * t;
        numerator / denominator
    }

    /// Calculate water sound speed as function of temperature
    /// Based on Bilaniuk & Wong (1993)
    pub fn sound_speed(temperature: f64) -> f64 {
        // Temperature in Celsius
        let t = temperature;

        // 5th order polynomial fit (Bilaniuk & Wong, 1993)
        Self::SOUND_SPEED_C0
            + Self::SOUND_SPEED_C1 * t
            + Self::SOUND_SPEED_C2 * t.powi(2)
            + Self::SOUND_SPEED_C3 * t.powi(3)
            + Self::SOUND_SPEED_C4 * t.powi(4)
            + Self::SOUND_SPEED_C5 * t.powi(5)
    }

    /// Calculate water absorption coefficient
    /// Based on Francois & Garrison (1982) model
    pub fn absorption_coefficient(
        frequency: f64,   // Hz
        temperature: f64, // Celsius
        depth: f64,       // meters
        salinity: f64,    // parts per thousand
        ph: f64,          // pH value
    ) -> f64 {
        let f = frequency / 1e3; // Convert to kHz
        let t = temperature;
        let s = salinity;
        let d = depth / 1e3; // Convert to km

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
        alpha_db_per_km * 0.1151 / 1e3 // Convert to Np/m
    }

    /// Pinkerton model for absorption calculations
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
        // Density could be used for more accurate calculation but simplified here
        // let density = Self::to_density(hu);

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
        theta: f64,      // Angle from axis (radians)
        radius: f64,     // Piston radius (m)
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
        theta_x: f64,    // Angle in x-plane (radians)
        theta_y: f64,    // Angle in y-plane (radians)
        width: f64,      // Width (m)
        height: f64,     // Height (m)
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
    pub fn apply_time_window(signal: &mut Array1<f64>, window_type: WindowType) {
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
        let mut complex_signal: Vec<Complex<f64>> =
            signal.iter().map(|&x| Complex::new(x, 0.0)).collect();

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
            complex_signal.iter().map(|c| c.re / n as f64).collect(),
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
        for i in 1..nx - 1 {
            for j in 0..ny {
                grad_x[[i, j]] = (field[[i + 1, j]] - field[[i - 1, j]]) / (2.0 * dx);
            }
        }
        // Forward/backward differences at boundaries
        for j in 0..ny {
            grad_x[[0, j]] = (field[[1, j]] - field[[0, j]]) / dx;
            grad_x[[nx - 1, j]] = (field[[nx - 1, j]] - field[[nx - 2, j]]) / dx;
        }

        // Y-gradient with central differences
        for i in 0..nx {
            for j in 1..ny - 1 {
                grad_y[[i, j]] = (field[[i, j + 1]] - field[[i, j - 1]]) / (2.0 * dy);
            }
        }
        // Forward/backward differences at boundaries
        for i in 0..nx {
            grad_y[[i, 0]] = (field[[i, 1]] - field[[i, 0]]) / dy;
            grad_y[[i, ny - 1]] = (field[[i, ny - 1]] - field[[i, ny - 2]]) / dy;
        }

        (grad_x, grad_y)
    }

    /// Calculate Laplacian using central differences
    ///
    /// Computes the discrete Laplacian (∇²) of a 2D field using finite differences.
    /// Interior points use standard central differences, while boundary points use
    /// forward/backward differences to maintain accuracy at edges.
    ///
    /// # Boundary Handling
    /// - Corners: Uses one-sided differences in both directions
    /// - Edges: Uses one-sided difference normal to edge, central difference along edge
    /// - Interior: Uses standard 5-point stencil with central differences
    ///
    /// For very small grids (< 3x3), uses first-order differencing schemes.
    pub fn laplacian_2d(field: &Array2<f64>, dx: f64, dy: f64) -> Array2<f64> {
        let (nx, ny) = field.dim();
        let mut laplacian = Array2::zeros((nx, ny));

        // Handle edge cases for very small grids
        if nx < 3 || ny < 3 {
            // For very small grids, use first-order finite differences
            if nx >= 2 && ny >= 2 {
                for i in 0..nx {
                    for j in 0..ny {
                        let mut lap_x = 0.0;
                        let mut lap_y = 0.0;

                        // X-direction
                        if i == 0 && nx > 1 {
                            lap_x = (field[[1, j]] - field[[0, j]]) / (dx * dx);
                        } else if i == nx - 1 && nx > 1 {
                            lap_x = (field[[nx - 2, j]] - field[[nx - 1, j]]) / (dx * dx);
                        } else if i > 0 && i < nx - 1 {
                            lap_x = (field[[i + 1, j]] - 2.0 * field[[i, j]] + field[[i - 1, j]])
                                / (dx * dx);
                        }

                        // Y-direction
                        if j == 0 && ny > 1 {
                            lap_y = (field[[i, 1]] - field[[i, 0]]) / (dy * dy);
                        } else if j == ny - 1 && ny > 1 {
                            lap_y = (field[[i, ny - 2]] - field[[i, ny - 1]]) / (dy * dy);
                        } else if j > 0 && j < ny - 1 {
                            lap_y = (field[[i, j + 1]] - 2.0 * field[[i, j]] + field[[i, j - 1]])
                                / (dy * dy);
                        }

                        laplacian[[i, j]] = lap_x + lap_y;
                    }
                }
            }
            return laplacian;
        }

        let dx2 = dx * dx;
        let dy2 = dy * dy;

        // Interior points: standard 5-point stencil
        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                laplacian[[i, j]] = (field[[i + 1, j]] - 2.0 * field[[i, j]] + field[[i - 1, j]])
                    / dx2
                    + (field[[i, j + 1]] - 2.0 * field[[i, j]] + field[[i, j - 1]]) / dy2;
            }
        }

        // Left and right boundaries (excluding corners)
        for j in 1..ny - 1 {
            // Left boundary (i=0): forward difference in x, central in y
            if nx >= 3 {
                laplacian[[0, j]] = (field[[2, j]] - 2.0 * field[[1, j]] + field[[0, j]]) / dx2
                    + (field[[0, j + 1]] - 2.0 * field[[0, j]] + field[[0, j - 1]]) / dy2;
            } else {
                laplacian[[0, j]] = (field[[1, j]] - field[[0, j]]) / dx2
                    + (field[[0, j + 1]] - 2.0 * field[[0, j]] + field[[0, j - 1]]) / dy2;
            }

            // Right boundary (i=nx-1): backward difference in x, central in y
            let i = nx - 1;
            if nx >= 3 {
                laplacian[[i, j]] = (field[[i, j]] - 2.0 * field[[i - 1, j]] + field[[i - 2, j]])
                    / dx2
                    + (field[[i, j + 1]] - 2.0 * field[[i, j]] + field[[i, j - 1]]) / dy2;
            } else {
                laplacian[[i, j]] = (field[[i - 1, j]] - field[[i, j]]) / dx2
                    + (field[[i, j + 1]] - 2.0 * field[[i, j]] + field[[i, j - 1]]) / dy2;
            }
        }

        // Top and bottom boundaries (excluding corners)
        for i in 1..nx - 1 {
            // Bottom boundary (j=0): central difference in x, forward in y
            if ny >= 3 {
                laplacian[[i, 0]] = (field[[i + 1, 0]] - 2.0 * field[[i, 0]] + field[[i - 1, 0]])
                    / dx2
                    + (field[[i, 2]] - 2.0 * field[[i, 1]] + field[[i, 0]]) / dy2;
            } else {
                laplacian[[i, 0]] = (field[[i + 1, 0]] - 2.0 * field[[i, 0]] + field[[i - 1, 0]])
                    / dx2
                    + (field[[i, 1]] - field[[i, 0]]) / dy2;
            }

            // Top boundary (j=ny-1): central difference in x, backward in y
            let j = ny - 1;
            if ny >= 3 {
                laplacian[[i, j]] = (field[[i + 1, j]] - 2.0 * field[[i, j]] + field[[i - 1, j]])
                    / dx2
                    + (field[[i, j]] - 2.0 * field[[i, j - 1]] + field[[i, j - 2]]) / dy2;
            } else {
                laplacian[[i, j]] = (field[[i + 1, j]] - 2.0 * field[[i, j]] + field[[i - 1, j]])
                    / dx2
                    + (field[[i, j - 1]] - field[[i, j]]) / dy2;
            }
        }

        // Corner points: use one-sided differences in both directions
        // Bottom-left corner (0, 0)
        if nx >= 3 && ny >= 3 {
            laplacian[[0, 0]] = (field[[2, 0]] - 2.0 * field[[1, 0]] + field[[0, 0]]) / dx2
                + (field[[0, 2]] - 2.0 * field[[0, 1]] + field[[0, 0]]) / dy2;
        } else {
            let lap_x = if nx >= 2 {
                (field[[1, 0]] - field[[0, 0]]) / dx2
            } else {
                0.0
            };
            let lap_y = if ny >= 2 {
                (field[[0, 1]] - field[[0, 0]]) / dy2
            } else {
                0.0
            };
            laplacian[[0, 0]] = lap_x + lap_y;
        }

        // Bottom-right corner (nx-1, 0)
        if nx >= 3 && ny >= 3 {
            laplacian[[nx - 1, 0]] =
                (field[[nx - 1, 0]] - 2.0 * field[[nx - 2, 0]] + field[[nx - 3, 0]]) / dx2
                    + (field[[nx - 1, 2]] - 2.0 * field[[nx - 1, 1]] + field[[nx - 1, 0]]) / dy2;
        } else {
            let lap_x = if nx >= 2 {
                (field[[nx - 2, 0]] - field[[nx - 1, 0]]) / dx2
            } else {
                0.0
            };
            let lap_y = if ny >= 2 {
                (field[[nx - 1, 1]] - field[[nx - 1, 0]]) / dy2
            } else {
                0.0
            };
            laplacian[[nx - 1, 0]] = lap_x + lap_y;
        }

        // Top-left corner (0, ny-1)
        if nx >= 3 && ny >= 3 {
            laplacian[[0, ny - 1]] =
                (field[[2, ny - 1]] - 2.0 * field[[1, ny - 1]] + field[[0, ny - 1]]) / dx2
                    + (field[[0, ny - 1]] - 2.0 * field[[0, ny - 2]] + field[[0, ny - 3]]) / dy2;
        } else {
            let lap_x = if nx >= 2 {
                (field[[1, ny - 1]] - field[[0, ny - 1]]) / dx2
            } else {
                0.0
            };
            let lap_y = if ny >= 2 {
                (field[[0, ny - 2]] - field[[0, ny - 1]]) / dy2
            } else {
                0.0
            };
            laplacian[[0, ny - 1]] = lap_x + lap_y;
        }

        // Top-right corner (nx-1, ny-1)
        if nx >= 3 && ny >= 3 {
            laplacian[[nx - 1, ny - 1]] = (field[[nx - 1, ny - 1]] - 2.0 * field[[nx - 2, ny - 1]]
                + field[[nx - 3, ny - 1]])
                / dx2
                + (field[[nx - 1, ny - 1]] - 2.0 * field[[nx - 1, ny - 2]]
                    + field[[nx - 1, ny - 3]])
                    / dy2;
        } else {
            let lap_x = if nx >= 2 {
                (field[[nx - 2, ny - 1]] - field[[nx - 1, ny - 1]]) / dx2
            } else {
                0.0
            };
            let lap_y = if ny >= 2 {
                (field[[nx - 1, ny - 2]] - field[[nx - 1, ny - 1]]) / dy2
            } else {
                0.0
            };
            laplacian[[nx - 1, ny - 1]] = lap_x + lap_y;
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
                        let ii = (i as i32 + ki as i32 - half_size as i32)
                            .max(0)
                            .min(nx as i32 - 1) as usize;
                        let jj = (j as i32 + kj as i32 - half_size as i32)
                            .max(0)
                            .min(ny as i32 - 1) as usize;
                        value += field[[ii, jj]] * kernel[[ki, kj]];
                    }
                }

                smoothed[[i, j]] = value;
            }
        }

        smoothed
    }
}

/// Bessel function J1 calculation
///
/// Approximates the Bessel function of the first kind, order one (J₁(x)), using:
/// - A polynomial expansion for |x| < 3.0
/// - An asymptotic expansion for |x| >= 3.0
///
/// # Numerical Accuracy
/// - For |x| < 3.0, the polynomial expansion provides reasonable accuracy for small arguments,
///   but may deviate from the true value as |x| approaches 3.0.
/// - For |x| >= 3.0, the asymptotic expansion is accurate for large arguments, but may lose
///   precision for moderately sized x (e.g., 3 < |x| < 5).
/// - Maximum relative error is typically less than 1% for |x| > 5, but can be higher near the transition.
///
/// # Valid Range
/// - Valid for all real x, but accuracy is best for |x| << 3.0 (polynomial) and |x| >> 3.0 (asymptotic).
/// - For |x| ≈ 3.0, accuracy may be reduced.
///
/// # Numerical Limitations
/// - For very large |x|, floating-point precision may degrade.
/// - For very small |x|, the polynomial expansion is stable.
/// - No special handling for NaN or infinite values.
///
/// # References
/// - Abramowitz & Stegun, "Handbook of Mathematical Functions", 9.1.21, 9.2.1
/// - Numerical Recipes, 6.5
fn bessel_j1(x: f64) -> f64 {
    // Polynomial expansion for small x
    if x.abs() < 3.0 {
        let x2 = x * x;
        x * (0.5 - x2 / 8.0 + x2 * x2 / 192.0 - x2 * x2 * x2 / 9216.0)
    } else {
        // Asymptotic expansion for large x
        // inv_x could be used for higher order terms but not needed here
        // let inv_x = 1.0 / x;
        let phase = x - 3.0 * PI / 4.0;
        (2.0 / (PI * x)).sqrt()
            * phase.cos()
            * (1.0 - 3.0 / (8.0 * x * x) + 15.0 / (128.0 * x * x * x * x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_laplacian_boundaries() {
        // Test that Laplacian handles boundaries correctly
        // Use a quadratic field where we know the analytical Laplacian
        let field = Array2::from_shape_fn((5, 5), |(i, j)| {
            let x = i as f64;
            let y = j as f64;
            x * x + y * y // f(x,y) = x² + y², so ∇²f = 2 + 2 = 4
        });

        let laplacian = NumericalUtils::laplacian_2d(&field, 1.0, 1.0);

        // Check that all points have been computed (non-zero for this field)
        // The Laplacian of x² + y² should be approximately 4 everywhere
        // (with some numerical error at boundaries due to one-sided differences)

        // Interior points should be close to 4
        for i in 1..4 {
            for j in 1..4 {
                assert_relative_eq!(laplacian[[i, j]], 4.0, epsilon = 0.1);
            }
        }

        // Boundaries should be computed (not left as zero)
        // They might not be exactly 4 due to one-sided differences, but should be non-zero
        assert!(
            laplacian[[0, 0]].abs() > 1.0,
            "Corner (0,0) should be computed"
        );
        assert!(
            laplacian[[0, 2]].abs() > 1.0,
            "Left edge (0,2) should be computed"
        );
        assert!(
            laplacian[[4, 2]].abs() > 1.0,
            "Right edge (4,2) should be computed"
        );
        assert!(
            laplacian[[2, 0]].abs() > 1.0,
            "Bottom edge (2,0) should be computed"
        );
        assert!(
            laplacian[[2, 4]].abs() > 1.0,
            "Top edge (2,4) should be computed"
        );

        // Test with a constant field (Laplacian should be zero everywhere)
        let const_field = Array2::ones((5, 5));
        let const_laplacian = NumericalUtils::laplacian_2d(&const_field, 1.0, 1.0);

        for value in const_laplacian.iter() {
            assert!(
                value.abs() < 1e-10,
                "Laplacian of constant field should be zero"
            );
        }

        // Test with small grid
        let small_field = Array2::from_shape_fn((2, 2), |(i, j)| (i + j) as f64);
        let small_laplacian = NumericalUtils::laplacian_2d(&small_field, 1.0, 1.0);
        assert_eq!(small_laplacian.dim(), (2, 2), "Should handle 2x2 grid");

        // Test that boundaries are actually computed, not left as zero
        // Use a linear field where boundaries matter
        let linear_field = Array2::from_shape_fn((4, 4), |(i, j)| {
            i as f64 + j as f64 // Linear field, Laplacian should be 0
        });
        let linear_laplacian = NumericalUtils::laplacian_2d(&linear_field, 1.0, 1.0);

        // All values should be close to zero for a linear field
        for value in linear_laplacian.iter() {
            assert!(value.abs() < 0.1, "Laplacian of linear field should be ~0");
        }
    }

    #[test]
    fn test_gradient_laplacian_consistency() {
        // Test that gradient and Laplacian are consistent
        // For a quadratic field f(x,y) = x² + y², ∇²f = 2 + 2 = 4
        let field = Array2::from_shape_fn((10, 10), |(i, j)| {
            let x = i as f64 * 0.1;
            let y = j as f64 * 0.1;
            x * x + y * y
        });

        let laplacian = NumericalUtils::laplacian_2d(&field, 0.1, 0.1);

        // Check interior points (should be close to 4)
        for i in 2..8 {
            for j in 2..8 {
                assert_relative_eq!(laplacian[[i, j]], 4.0, epsilon = 0.01);
            }
        }
    }

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
