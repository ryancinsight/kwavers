//! Multi-angle Plane Wave Compounding
//!
//! ## Algorithm
//!
//! 1. Transmit plane waves at multiple angles
//! 2. Reconstruct individual images for each angle
//! 3. Coherently compound the images
//!
//! ## References
//!
//! - Montaldo et al. (2009), "Coherent plane-wave compounding"

use ndarray::{Array2, Array3};
use super::plane_wave::PlaneWaveConfig;

/// Multi-angle plane wave compounding
#[derive(Debug)]
pub struct PlaneWaveCompounding {
    #[allow(dead_code)]
    configs: Vec<PlaneWaveConfig>,
}

impl PlaneWaveCompounding {
    /// Create new compounding processor
    #[must_use]
    pub fn new(angles: &[f64], base_config: PlaneWaveConfig) -> Self {
        let configs = angles
            .iter()
            .map(|&angle| PlaneWaveConfig {
                tx_angle: angle,
                ..base_config.clone()
            })
            .collect();

        Self { configs }
    }

    /// Compound multiple plane wave images
    #[must_use]
    pub fn compound(&self, images: &Array3<f64>) -> Array2<f64> {
        let (num_angles, height, width) = images.dim();
        let mut compounded = Array2::<f64>::zeros((height, width));

        for i in 0..height {
            for j in 0..width {
                let mut sum = 0.0;
                for angle in 0..num_angles {
                    sum += images[[angle, i, j]];
                }
                compounded[[i, j]] = sum / num_angles as f64;
            }
        }

        compounded
    }
}
