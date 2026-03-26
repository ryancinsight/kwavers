//! Synthetic Aperture (SA) Imaging
//!
//! ## Algorithm
//!
//! 1. For each transmit element, compute round-trip delays to image points
//! 2. Apply phase correction based on transmit and receive delays
//! 3. Coherently sum contributions from all transmit-receive pairs
//! 4. Apply apodization and envelope detection
//!
//! ## References
//!
//! - Karaman et al. (1995), "Synthetic aperture imaging for small scale systems"
//! - Jensen et al. (2006), "Synthetic aperture ultrasound imaging"

use ndarray::{Array2, Array3};
use num_complex::Complex64;
use std::f64::consts::PI;

/// Synthetic Aperture (SA) imaging configuration
#[derive(Debug, Clone)]
pub struct SyntheticApertureConfig {
    /// Number of transmit elements
    pub num_tx_elements: usize,
    /// Number of receive elements
    pub num_rx_elements: usize,
    /// Element spacing (m)
    pub element_spacing: f64,
    /// Sound speed (m/s)
    pub sound_speed: f64,
    /// Center frequency (Hz)
    pub frequency: f64,
    /// Sampling frequency (Hz)
    pub sampling_frequency: f64,
    /// Number of transmit angles for compound imaging
    pub num_tx_angles: usize,
}

impl Default for SyntheticApertureConfig {
    fn default() -> Self {
        Self {
            num_tx_elements: 64,
            num_rx_elements: 64,
            element_spacing: 0.3e-3,
            sound_speed: 1540.0,
            frequency: 5e6,
            sampling_frequency: 40e6,
            num_tx_angles: 1,
        }
    }
}

/// Synthetic Aperture reconstruction
#[derive(Debug)]
pub struct SyntheticApertureReconstruction {
    config: SyntheticApertureConfig,
}

impl SyntheticApertureReconstruction {
    /// Create new SA reconstruction
    #[must_use]
    pub fn new(config: SyntheticApertureConfig) -> Self {
        Self { config }
    }

    /// Reconstruct SA image from RF data
    #[must_use]
    pub fn reconstruct(&self, rf_data: &Array3<f64>, image_grid: &Array3<f64>) -> Array2<f64> {
        let (n_samples, n_rx, n_tx) = rf_data.dim();
        let (_, height, width) = image_grid.dim();

        let mut image = Array2::<f64>::zeros((height, width));

        for i in 0..height {
            for j in 0..width {
                let x = image_grid[[0, i, j]];
                let z = image_grid[[1, i, j]];

                let mut sum = Complex64::new(0.0, 0.0);

                for tx in 0..n_tx {
                    for rx in 0..n_rx {
                        let tx_x =
                            (tx as f64 - (n_tx - 1) as f64 / 2.0) * self.config.element_spacing;
                        let tx_delay = self.calculate_delay(tx_x, 0.0, x, z);

                        let rx_x =
                            (rx as f64 - (n_rx - 1) as f64 / 2.0) * self.config.element_spacing;
                        let rx_delay = self.calculate_delay(x, z, rx_x, 0.0);

                        let total_delay = tx_delay + rx_delay;

                        let sample_idx = (total_delay * self.config.sampling_frequency) as usize;
                        if sample_idx < n_samples {
                            let rf_sample = rf_data[[sample_idx, rx, tx]];

                            let phase_correction = Complex64::new(
                                0.0,
                                -2.0 * PI * self.config.frequency * total_delay,
                            )
                            .exp();

                            sum += rf_sample * phase_correction;
                        }
                    }
                }

                image[[i, j]] = sum.norm();
            }
        }

        image
    }

    /// Calculate propagation delay between two points
    fn calculate_delay(&self, x1: f64, z1: f64, x2: f64, z2: f64) -> f64 {
        let distance = ((x2 - x1).powi(2) + (z2 - z1).powi(2)).sqrt();
        distance / self.config.sound_speed
    }
}
