//! Skull attenuation modeling
//!
//! Reference: Pinton et al. (2012) "Attenuation, scattering, and absorption
//! of ultrasound in the skull bone"

use crate::error::KwaversResult;
use crate::grid::Grid;
use ndarray::Array3;

/// Skull attenuation calculator
#[derive(Debug)]
pub struct SkullAttenuation {
    /// Base attenuation coefficient (Np/m/MHz)
    alpha_0: f64,
    /// Power law exponent (typically 1.0 for bone)
    exponent: f64,
}

impl Default for SkullAttenuation {
    fn default() -> Self {
        Self {
            alpha_0: 60.0, // Np/m/MHz for cortical bone
            exponent: 1.0,
        }
    }
}

impl SkullAttenuation {
    /// Create new attenuation calculator
    pub fn new(alpha_0: f64, exponent: f64) -> Self {
        Self { alpha_0, exponent }
    }

    /// Compute attenuation factor for frequency
    pub fn compute_attenuation_field(
        &self,
        grid: &Grid,
        skull_mask: &Array3<f64>,
        frequency: f64,
    ) -> KwaversResult<Array3<f64>> {
        let freq_mhz = frequency / 1e6;
        let alpha = self.alpha_0 * freq_mhz.powf(self.exponent);

        let mut attenuation = Array3::ones(skull_mask.dim());

        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    if skull_mask[[i, j, k]] > 0.5 {
                        // Path length through skull (simplified)
                        let path_length = grid.dx;
                        attenuation[[i, j, k]] = (-alpha * path_length).exp();
                    }
                }
            }
        }

        Ok(attenuation)
    }
}
