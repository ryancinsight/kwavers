//! Power-law absorption model

use kwavers_core::constants::acoustic_parameters::{ABSORPTION_TISSUE, WATER_ABSORPTION_ALPHA_0};
use kwavers_core::constants::numerical::{CM_TO_M, MHZ_TO_HZ, TWO_PI};
use kwavers_core::constants::{DB_TO_NP, REFERENCE_FREQUENCY_HZ};
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
            alpha_0: ABSORPTION_TISSUE, // 0.75 dB/(MHz^y·cm) — Duck (1990) Table 4.1
            y: 1.1,        // Typical soft tissue exponent
            f_ref: REFERENCE_FREQUENCY_HZ,
            dispersion_correction: true,
        }
    }
}

impl PowerLawAbsorption {
    /// Create absorption model for water at 20°C
    #[must_use]
    pub fn water() -> Self {
        Self {
            alpha_0: WATER_ABSORPTION_ALPHA_0, // 0.0022 dB/(MHz^y·cm) — Duck (1990) Table 4.1
            y: 2.0,          // Quadratic frequency dependence
            f_ref: REFERENCE_FREQUENCY_HZ,
            dispersion_correction: false,
        }
    }

    /// Create absorption model for soft tissue
    #[must_use]
    pub fn soft_tissue() -> Self {
        Self {
            alpha_0: ABSORPTION_TISSUE,
            y: 1.1,
            f_ref: REFERENCE_FREQUENCY_HZ,
            dispersion_correction: true,
        }
    }

    /// Calculate absorption coefficient at given frequency
    #[must_use]
    pub fn absorption_at_frequency(&self, frequency: f64) -> f64 {
        // Convert dB/(MHz^y cm) → Np/m:
        //   α [Np/m] = α [dB/cm] · (1/CM_TO_M) [cm/m] · DB_TO_NP [Np/dB]
        let f_mhz = frequency / MHZ_TO_HZ;
        let alpha_db = self.alpha_0 * f_mhz.powf(self.y);

        alpha_db * DB_TO_NP / CM_TO_M
    }

    /// Phase velocity from the Szabo / Treeby-Cox Kramers-Kronig relation
    /// for power-law absorption.
    ///
    /// ## Formula (Szabo 1994 Eq 13; Treeby & Cox 2010 Eq 11)
    ///
    /// ```text
    /// c_p(ω)⁻¹ − c_0⁻¹ = α(ω) · tan(πy/2) / ω
    /// ```
    ///
    /// where `α(ω) = α(ω)` (Np/m) is obtained from
    /// [`Self::absorption_at_frequency`] (which handles the dB/(cm·MHz^y) →
    /// Np/m conversion).  Inverting:
    ///
    /// ```text
    /// c_p(ω) = c_0 / (1 + c_0 · α(ω) · tan(πy/2) / ω)
    /// ```
    ///
    /// Prior to 2026-05-21 this used `self.alpha_0` directly in the
    /// formula — a value stored in dB/(cm·MHz^y), not the SI Np/m
    /// required by Kramers-Kronig — *and* omitted the `c_0` factor that
    /// scales the inverse-phase-velocity correction.  Combined, the prior
    /// dispersion correction was off by ~10⁵× from the canonical result.
    ///
    /// ## References
    /// - Szabo T. L. (1994). *J. Acoust. Soc. Am.* 96(1), 491–500. Eq 13.
    /// - Treeby B. E. & Cox B. T. (2010). *J. Acoust. Soc. Am.* 127(5), 2741. Eq 11.
    #[must_use]
    pub fn phase_velocity(&self, frequency: f64, c0: f64) -> f64 {
        if !self.dispersion_correction || frequency == 0.0 {
            return c0;
        }

        let omega = TWO_PI * frequency;
        let tan_term = (std::f64::consts::PI * self.y / 2.0).tan();
        let alpha = self.absorption_at_frequency(frequency); // Np/m

        let denom = 1.0 + c0 * alpha * tan_term / omega;
        c0 / denom
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
                    slice.par_mapv_inplace(|c| c * attenuation);
                }
            });
    }
}
