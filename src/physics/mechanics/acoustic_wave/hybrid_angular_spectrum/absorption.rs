//! Absorption operator with power law model
//!
//! Reference: Szabo (1994) "Time domain wave equations for lossy media"

use crate::error::KwaversResult;
use crate::physics::mechanics::acoustic_wave::hybrid_angular_spectrum::HASConfig;
use ndarray::Array3;

/// Absorption operator for power law attenuation
#[derive(Debug)]
pub struct AbsorptionOperator {
    attenuation_coeff: f64,
    power_law_exp: f64,
    reference_freq: f64,
}

impl AbsorptionOperator {
    /// Create new absorption operator
    pub fn new(config: &HASConfig) -> Self {
        Self {
            attenuation_coeff: config.attenuation_coeff,
            power_law_exp: config.power_law_exponent,
            reference_freq: config.reference_frequency,
        }
    }

    /// Apply absorption step
    ///
    /// Applies exponential decay: p(z + Δz) = p(z) × exp(-α(f) × Δz)
    pub fn apply(&self, pressure: &Array3<f64>, dz: f64) -> KwaversResult<Array3<f64>> {
        // Calculate attenuation at reference frequency
        let freq_mhz = self.reference_freq / 1e6;
        let alpha_db_cm = self.attenuation_coeff * freq_mhz.powf(self.power_law_exp);
        let alpha_np_m = alpha_db_cm * 100.0 / 8.686; // dB/cm to Np/m

        // Apply exponential attenuation
        let factor = (-alpha_np_m * dz).exp();
        let result = pressure.mapv(|p| p * factor);

        Ok(result)
    }
}
