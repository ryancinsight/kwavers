//! Skull attenuation modeling
//!
//! Reference: Pinton et al. (2012) "Attenuation, scattering, and absorption
//! of ultrasound in the skull bone"

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use ndarray::Array3;

/// Skull attenuation calculator
/// TODO_AUDIT: P1 - Advanced Skull Attenuation - Implement frequency-dependent attenuation with scattering and dispersion effects for accurate transcranial propagation
/// DEPENDS ON: physics/acoustics/skull/attenuation/scattering.rs, physics/acoustics/skull/attenuation/dispersion.rs, physics/acoustics/skull/attenuation/temperature.rs
/// MISSING: Scattering losses due to trabecular microstructure
/// MISSING: Frequency dispersion in cancellous vs cortical bone
/// MISSING: Temperature-dependent attenuation changes
/// MISSING: Anisotropic attenuation due to preferred orientations
/// MISSING: Non-linear attenuation at high intensities
/// THEOREM: Kramers-Kronig dispersion: ε''(ω) ∝ ∫ ε'(ω')/(ω'-ω) dω' for causality
/// THEOREM: Multiple scattering: α_total = α_absorption + α_scattering with interference effects
/// REFERENCES: Pinton et al. (2012) JASA 131, 4694; White et al. (2006) Ultrasound Med Biol
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
                        // Path length through skull bone layer
                        let path_length = grid.dx;
                        attenuation[[i, j, k]] = (-alpha * path_length).exp();
                    }
                }
            }
        }

        Ok(attenuation)
    }
}
