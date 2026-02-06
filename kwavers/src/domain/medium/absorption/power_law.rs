//! Power-law absorption model

use serde::{Deserialize, Serialize};

/// Power-law absorption model configuration
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PowerLawAbsorption {
    /// Absorption coefficient at 1 `MHz` [dB/(MHz^y cm)]
    pub alpha_0: f64,
    /// Power law exponent (typically 1.0-1.5)
    pub y: f64,
    /// Reference frequency for `alpha_0` \[Hz\]
    pub f_ref: f64,
    /// Enable dispersion correction
    pub dispersion_correction: bool,
}

impl Default for PowerLawAbsorption {
    fn default() -> Self {
        Self {
            alpha_0: 0.75, // Typical soft tissue value
            y: 1.1,        // Typical soft tissue exponent
            f_ref: 1e6,    // 1 MHz reference
            dispersion_correction: true,
        }
    }
}

impl PowerLawAbsorption {
    /// Create absorption model for water at 20°C
    #[must_use]
    pub fn water() -> Self {
        Self {
            alpha_0: 0.0022, // dB/(MHz^2 cm) for water
            y: 2.0,          // Quadratic frequency dependence
            f_ref: 1e6,
            dispersion_correction: false,
        }
    }

    /// Create absorption model for soft tissue
    #[must_use]
    pub fn soft_tissue() -> Self {
        Self {
            alpha_0: 0.75,
            y: 1.1,
            f_ref: 1e6,
            dispersion_correction: true,
        }
    }

    /// Calculate absorption coefficient at given frequency
    #[must_use]
    pub fn absorption_at_frequency(&self, frequency: f64) -> f64 {
        // Convert to Np/m from dB/(MHz^y cm)
        const DB_TO_NP: f64 = 1.0 / 8.686; // 1 Np = 8.686 dB
        const CM_TO_M: f64 = 100.0;
        const MHZ_TO_HZ: f64 = 1e6;

        let f_mhz = frequency / MHZ_TO_HZ;
        let alpha_db = self.alpha_0 * f_mhz.powf(self.y);

        alpha_db * DB_TO_NP * CM_TO_M
    }

    /// Calculate phase velocity from absorption (Kramers-Kronig relations)
    #[must_use]
    pub fn phase_velocity(&self, frequency: f64, c0: f64) -> f64 {
        if !self.dispersion_correction {
            return c0;
        }

        // Kramers-Kronig relation for power law absorption
        // c(ω) = c₀ / (1 - α₀ * tan(πy/2) * ω^(y-1))
        let omega = 2.0 * std::f64::consts::PI * frequency;
        let tan_term = (std::f64::consts::PI * self.y / 2.0).tan();

        c0 / (1.0 + self.alpha_0 * tan_term * omega.powf(self.y - 1.0))
    }
}

/// Power law model implementation
#[derive(Debug)]
pub struct PowerLawModel {
    config: PowerLawAbsorption,
}

impl PowerLawModel {
    /// Create a new power law model
    #[must_use]
    pub fn new(config: PowerLawAbsorption) -> Self {
        Self { config }
    }

    /// Apply power law absorption in frequency domain
    pub fn apply_frequency_domain(
        &self,
        spectrum: &mut ndarray::Array3<num_complex::Complex<f64>>,
        frequencies: &[f64],
        distance: f64,
    ) {
        use ndarray::Zip;

        Zip::from(spectrum.outer_iter_mut())
            .and(frequencies)
            .for_each(|mut slice, &freq| {
                if freq > 0.0 {
                    let alpha = self.config.absorption_at_frequency(freq);
                    let attenuation = (-alpha * distance).exp();
                    slice.mapv_inplace(|c| c * attenuation);
                }
            });
    }
}
