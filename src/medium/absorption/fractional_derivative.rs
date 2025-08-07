//! Fractional derivative absorption model for biological tissues
//! 
//! This module implements fractional derivative models for acoustic absorption
//! in biological tissues, which provide more accurate frequency-dependent
//! attenuation than traditional integer-order models.
//!
//! References:
//! - Szabo, T. L. (1994). "Time domain wave equations for lossy media
//!   obeying a frequency power law" Journal of the Acoustical Society
//!   of America, 96(1), 491-500.
//! - Treeby, B. E., & Cox, B. T. (2010). "Modeling power law absorption
//!   and dispersion for acoustic propagation using the fractional Laplacian"
//!   Journal of the Acoustical Society of America, 127(5), 2741-2748.

use crate::{KwaversResult, KwaversError, ValidationError};
use crate::Grid;
use ndarray::{Array3, Array1, Zip};
use rustfft::{FftPlanner, num_complex::Complex};
use std::collections::VecDeque;
use std::f64::consts::PI;

/// Fractional derivative absorption model
#[derive(Debug, Clone)]
pub struct FractionalDerivativeAbsorption {
    /// Power law exponent (typically 1.0-2.0 for tissues)
    pub power_law_exponent: f64,
    /// Absorption coefficient at 1 MHz (Np/m)
    pub alpha_0: f64,
    /// Reference frequency (Hz)
    pub reference_frequency: f64,
    /// Fractional Laplacian order (related to power law exponent)
    pub fractional_order: f64,
    /// Memory variables for time-domain implementation
    memory_variables: Option<MemoryVariables>,
}

/// Memory variables for time-domain fractional derivative
#[derive(Debug, Clone)]
struct MemoryVariables {
    /// Previous pressure values for convolution (newest at front)
    pressure_history: VecDeque<Array3<f64>>,
    /// Convolution weights
    weights: Vec<f64>,
    /// Maximum history length
    max_history: usize,
}

impl FractionalDerivativeAbsorption {
    /// Create a new fractional derivative absorption model
    pub fn new(power_law_exponent: f64, alpha_0: f64, reference_frequency: f64) -> Self {
        // Fractional order is related to power law by: y = 2α - 1
        let fractional_order = power_law_exponent - 1.0;
        
        Self {
            power_law_exponent,
            alpha_0,
            reference_frequency,
            fractional_order,
            memory_variables: None,
        }
    }
    
    /// Initialize memory variables for time-domain implementation
    pub fn initialize_memory(&mut self, grid: &Grid, dt: f64, history_length: usize) -> KwaversResult<()> {
        // Validate parameters
        if self.power_law_exponent < 0.0 || self.power_law_exponent > 3.0 {
            return Err(KwaversError::Validation(ValidationError::RangeValidation {
                field: "power_law_exponent".to_string(),
                value: self.power_law_exponent.to_string(),
                min: 0.0,
                max: 3.0,
            }));
        }
        
        // Compute convolution weights using Grünwald-Letnikov approximation
        let weights = self.compute_grunwald_weights(history_length, dt)?;
        
        // Initialize pressure history as VecDeque with capacity
        let mut pressure_history = VecDeque::with_capacity(history_length);
        for _ in 0..history_length {
            pressure_history.push_back(grid.zeros_array());
        }
        
        self.memory_variables = Some(MemoryVariables {
            pressure_history,
            weights,
            max_history: history_length,
        });
        
        Ok(())
    }
    
    /// Compute Grünwald-Letnikov weights for fractional derivative
    fn compute_grunwald_weights(&self, n: usize, dt: f64) -> KwaversResult<Vec<f64>> {
        let alpha = self.fractional_order;
        let mut weights = vec![0.0; n];
        
        // First weight
        weights[0] = 1.0;
        
        // Recursive computation of weights
        for k in 1..n {
            weights[k] = weights[k-1] * (k as f64 - 1.0 - alpha) / k as f64;
        }
        
        // Scale by time step
        let dt_alpha = dt.powf(-alpha);
        weights.iter_mut().for_each(|w| *w *= dt_alpha);
        
        Ok(weights)
    }
    
    /// Apply fractional derivative absorption in time domain
    pub fn apply_absorption_time_domain(
        &mut self,
        pressure: &mut Array3<f64>,
        grid: &Grid,
        dt: f64,
    ) -> KwaversResult<()> {
        let memory = self.memory_variables.as_mut()
            .ok_or_else(|| KwaversError::Validation(ValidationError::FieldValidation {
                field: "memory_variables".to_string(),
                value: "None".to_string(),
                constraint: "Must initialize memory variables first".to_string(),
            }))?;
        
        // Compute fractional Laplacian using convolution
        let mut fractional_laplacian = grid.zeros_array();
        
        // Apply convolution with history
        memory.pressure_history.iter()
            .zip(memory.weights.iter())
            .for_each(|(hist, &weight)| {
                Zip::from(&mut fractional_laplacian)
                    .and(hist)
                    .for_each(|fl, &h| *fl += weight * h);
            });
        
        // Apply absorption
        let absorption_factor = -self.alpha_0 * self.reference_frequency.powf(2.0 - self.power_law_exponent);
        
        Zip::from(pressure)
            .and(&fractional_laplacian)
            .for_each(|p, &fl| {
                *p += absorption_factor * fl * dt;
            });
        
        // Update history efficiently with VecDeque
        // This is O(1) for both operations, compared to O(n) for Vec::insert(0)
        memory.pressure_history.pop_back();  // Remove oldest
        memory.pressure_history.push_front(pressure.clone());  // Add newest
        
        Ok(())
    }
    
    /// Apply fractional derivative absorption in frequency domain
    pub fn apply_absorption_frequency_domain(
        &self,
        pressure_spectrum: &mut Array3<Complex<f64>>,
        k_vec: &Array3<f64>,
        c0: f64,
        dt: f64,
    ) -> KwaversResult<()> {
        let (nx, ny, nz) = pressure_spectrum.dim();
        
        // Apply frequency-dependent absorption
        // For power law absorption: α(f) = α₀ * (f/f_ref)^y
        // The attenuation over time dt is: exp(-α(f) * c₀ * dt)
        
        (0..nx).for_each(|i| {
            (0..ny).for_each(|j| {
                (0..nz).for_each(|k| {
                    let k_mag = k_vec[[i, j, k]];
                    if k_mag > 0.0 {
                        // Convert wavenumber to frequency: f = k * c₀ / (2π)
                        let frequency = k_mag * c0 / (2.0 * PI);
                        
                        // Power law absorption coefficient
                        let alpha = self.alpha_0 * (frequency / self.reference_frequency)
                            .powf(self.power_law_exponent);
                        
                        // Apply absorption as exponential decay
                        // exp(-α * c₀ * dt) for propagation over distance c₀ * dt
                        let decay = (-alpha * c0 * dt).exp();
                        pressure_spectrum[[i, j, k]] *= decay;
                    }
                });
            });
        });
        
        Ok(())
    }
    
    /// Get frequency-dependent absorption coefficient
    pub fn absorption_coefficient(&self, frequency: f64) -> f64 {
        self.alpha_0 * (frequency / self.reference_frequency).powf(self.power_law_exponent)
    }
    
    /// Get frequency-dependent phase velocity
    pub fn phase_velocity(&self, frequency: f64, c0: f64) -> f64 {
        // Kramers-Kronig relations give dispersion from absorption
        let alpha = self.absorption_coefficient(frequency);
        let tan_term = (PI * self.power_law_exponent / 2.0).tan();
        c0 / (1.0 + alpha * c0 * tan_term / (2.0 * PI * frequency))
    }
}

/// Tissue-specific fractional derivative parameters
#[derive(Debug, Clone)]
pub struct TissueFractionalParameters {
    /// Tissue type name
    pub tissue_type: String,
    /// Power law exponent
    pub power_law: f64,
    /// Absorption at 1 MHz (dB/cm)
    pub alpha_0_db_cm: f64,
    /// Minimum valid frequency (Hz)
    pub min_frequency: f64,
    /// Maximum valid frequency (Hz)
    pub max_frequency: f64,
}

impl TissueFractionalParameters {
    /// Get parameters for common tissue types
    pub fn get_tissue_parameters(tissue: &str) -> Option<Self> {
        match tissue.to_lowercase().as_str() {
            "liver" => Some(Self {
                tissue_type: "liver".to_string(),
                power_law: 1.1,
                alpha_0_db_cm: 0.5,
                min_frequency: 0.5e6,
                max_frequency: 15e6,
            }),
            "breast" => Some(Self {
                tissue_type: "breast".to_string(),
                power_law: 1.5,
                alpha_0_db_cm: 0.75,
                min_frequency: 0.5e6,
                max_frequency: 15e6,
            }),
            "brain" => Some(Self {
                tissue_type: "brain".to_string(),
                power_law: 1.2,
                alpha_0_db_cm: 0.6,
                min_frequency: 0.5e6,
                max_frequency: 10e6,
            }),
            "muscle" => Some(Self {
                tissue_type: "muscle".to_string(),
                power_law: 1.1,
                alpha_0_db_cm: 1.0,
                min_frequency: 0.5e6,
                max_frequency: 15e6,
            }),
            "fat" => Some(Self {
                tissue_type: "fat".to_string(),
                power_law: 1.4,
                alpha_0_db_cm: 0.4,
                min_frequency: 0.5e6,
                max_frequency: 15e6,
            }),
            _ => None,
        }
    }
    
    /// Convert to FractionalDerivativeAbsorption model
    pub fn to_absorption_model(&self) -> FractionalDerivativeAbsorption {
        // Convert dB/cm to Np/m
        let alpha_0_np_m = self.alpha_0_db_cm * 100.0 / 8.686;
        
        FractionalDerivativeAbsorption::new(
            self.power_law,
            alpha_0_np_m,
            1e6, // Reference frequency of 1 MHz
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tissue_parameters() {
        let liver = TissueFractionalParameters::get_tissue_parameters("liver").unwrap();
        assert_eq!(liver.power_law, 1.1);
        assert_eq!(liver.tissue_type, "liver");
        
        let model = liver.to_absorption_model();
        assert!((model.power_law_exponent - 1.1).abs() < 1e-10);
    }
    
    #[test]
    fn test_frequency_dependent_absorption() {
        let model = FractionalDerivativeAbsorption::new(1.1, 0.5, 1e6);
        
        // Test at reference frequency
        let alpha_1mhz = model.absorption_coefficient(1e6);
        assert!((alpha_1mhz - 0.5).abs() < 1e-10);
        
        // Test power law scaling
        let alpha_2mhz = model.absorption_coefficient(2e6);
        let expected = 0.5 * 2.0_f64.powf(1.1);
        assert!((alpha_2mhz - expected).abs() < 1e-6);
    }
    
    #[test]
    fn test_frequency_domain_absorption() {
        use rustfft::num_complex::Complex;
        let model = FractionalDerivativeAbsorption::new(1.0, 0.1, 1e6);
        
        // Create test spectrum
        let mut spectrum = Array3::from_elem((3, 3, 3), Complex::new(1.0, 0.0));
        let mut k_vec = Array3::zeros((3, 3, 3));
        
        // Set a specific wavenumber
        let c0 = 1500.0;
        let freq = 2e6;
        k_vec[[1, 1, 1]] = 2.0 * PI * freq / c0;
        
        // Apply absorption
        let dt = 1e-6;
        model.apply_absorption_frequency_domain(&mut spectrum, &k_vec, c0, dt).unwrap();
        
        // Check that the center point was attenuated
        let expected_alpha = 0.1 * (freq / 1e6).powf(1.0);
        let expected_decay = (-expected_alpha * c0 * dt).exp();
        
        assert!((spectrum[[1, 1, 1]].re - expected_decay).abs() < 1e-10);
        assert!(spectrum[[1, 1, 1]].im.abs() < 1e-10);
        
        // Check that zero-frequency points were not attenuated
        assert!((spectrum[[0, 0, 0]].re - 1.0).abs() < 1e-10);
    }
}