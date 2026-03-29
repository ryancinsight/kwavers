//! Plane Wave Imaging (PWI) Reconstruction
//!
//! ## Algorithm
//!
//! 1. Transmit steered plane wave at angle θ
//! 2. Receive on all elements simultaneously
//! 3. For each image point, compute receive delays
//! 4. Apply phase correction and coherent summation
//!
//! ## References
//!
//! - Montaldo et al. (2009), "Coherent plane-wave compounding"

use crate::core::constants::fundamental::SOUND_SPEED_TISSUE;
use ndarray::{Array2, Array3};
use num_complex::Complex64;
use std::f64::consts::PI;

/// Plane wave imaging configuration
#[derive(Debug, Clone)]
pub struct PlaneWaveConfig {
    /// Transmit angle (radians)
    pub tx_angle: f64,
    /// Number of elements
    pub num_elements: usize,
    /// Element spacing (m)
    pub element_spacing: f64,
    /// Sound speed (m/s)
    pub sound_speed: f64,
    /// Center frequency (Hz)
    pub frequency: f64,
    /// Sampling frequency (Hz)
    pub sampling_frequency: f64,
}

impl Default for PlaneWaveConfig {
    fn default() -> Self {
        Self {
            tx_angle: 0.0,
            num_elements: 64,
            element_spacing: 0.3e-3,
            sound_speed: SOUND_SPEED_TISSUE,
            frequency: 5e6,
            sampling_frequency: 40e6,
        }
    }
}

/// Plane Wave Imaging reconstruction
#[derive(Debug)]
pub struct PlaneWaveReconstruction {
    config: PlaneWaveConfig,
}

impl PlaneWaveReconstruction {
    /// Create new PWI reconstruction
    #[must_use]
    pub fn new(config: PlaneWaveConfig) -> Self {
        Self { config }
    }

    /// Reconstruct PWI image from RF data
    #[must_use]
    pub fn reconstruct(&self, rf_data: &Array2<f64>, image_grid: &Array3<f64>) -> Array2<f64> {
        let (n_samples, n_elements) = rf_data.dim();
        let (_, height, width) = image_grid.dim();

        let mut image = Array2::<f64>::zeros((height, width));

        for i in 0..height {
            for j in 0..width {
                let x = image_grid[[0, i, j]];
                let z = image_grid[[1, i, j]];

                let mut sum = Complex64::new(0.0, 0.0);

                for elem in 0..n_elements {
                    let elem_x =
                        (elem as f64 - (n_elements - 1) as f64 / 2.0) * self.config.element_spacing;
                    let rx_delay = self.calculate_receive_delay(x, z, elem_x);

                    let sample_idx = (rx_delay * self.config.sampling_frequency) as usize;
                    if sample_idx < n_samples {
                        let rf_sample = rf_data[[sample_idx, elem]];

                        let phase_correction =
                            Complex64::new(0.0, -2.0 * PI * self.config.frequency * rx_delay).exp();

                        sum += rf_sample * phase_correction;
                    }
                }

                image[[i, j]] = sum.norm();
            }
        }

        image
    }

    /// Calculate receive delay for plane wave
    fn calculate_receive_delay(&self, x: f64, z: f64, elem_x: f64) -> f64 {
        let distance = ((x - elem_x).powi(2) + z.powi(2)).sqrt();
        distance / self.config.sound_speed
    }
}
