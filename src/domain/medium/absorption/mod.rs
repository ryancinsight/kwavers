//! Absorption and dispersion models for acoustic wave propagation
//!
//! This module implements state-of-the-art absorption models including:
//! - Power-law frequency dependence
//! - Fractional derivative formulations
//! - Tissue-specific absorption coefficients
//!
//! # Theory
//!
//! Acoustic absorption in biological tissues and other media follows a
//! power-law frequency dependence:
//! ```text
//! α(f) = α₀ * f^y
//! ```
//! where:
//! - α is the absorption coefficient \[Np/m\]
//! - α₀ is the absorption coefficient at 1 `MHz`
//! - f is frequency \[MHz\]
//! - y is the power law exponent (typically 1.0-1.5 for tissues)
//!
//! # References
//!
//! - Treeby, B. E., & Cox, B. T. (2010). "Modeling power law absorption and
//!   dispersion for acoustic propagation using the fractional Laplacian."
//!   JASA, 127(5), 2741-2748.
//! - Szabo, T. L. (2004). "Diagnostic ultrasound imaging: inside out."
//!   Academic Press.

pub mod dispersion;
pub mod fractional;
pub mod power_law;
pub mod stokes;
pub mod tissue;

pub use dispersion::{DispersionCorrection, DispersionModel};
pub use fractional::{FractionalDerivative, FractionalLaplacian};
pub use power_law::{PowerLawAbsorption, PowerLawModel};
pub use stokes::{StokesAbsorption, StokesParameters};
pub use tissue::{TissueAbsorption, TissueType, TISSUE_PROPERTIES};

use crate::domain::core::error::KwaversResult;
use ndarray::Array3;

/// Main absorption calculator that orchestrates different models
#[derive(Debug)]
pub struct AbsorptionCalculator {
    model: AbsorptionModel,
}

/// Absorption model selection
#[derive(Debug, Clone)]
pub enum AbsorptionModel {
    /// Power law absorption
    PowerLaw(PowerLawAbsorption),
    /// Tissue-specific absorption
    Tissue(TissueAbsorption),
    /// Stokes absorption for fluids
    Stokes(StokesAbsorption),
    /// Fractional Laplacian model
    Fractional(FractionalLaplacian),
}

impl AbsorptionCalculator {
    /// Create a new absorption calculator
    #[must_use]
    pub fn new(model: AbsorptionModel) -> Self {
        Self { model }
    }

    /// Calculate absorption coefficient at a given frequency
    #[must_use]
    pub fn absorption_coefficient(&self, frequency: f64) -> f64 {
        match &self.model {
            AbsorptionModel::PowerLaw(m) => m.absorption_at_frequency(frequency),
            AbsorptionModel::Tissue(m) => m.absorption_at_frequency(frequency),
            AbsorptionModel::Stokes(m) => m.absorption_at_frequency(frequency),
            AbsorptionModel::Fractional(m) => m.absorption_at_frequency(frequency),
        }
    }

    /// Apply absorption to a field
    pub fn apply_absorption(
        &self,
        field: &mut Array3<f64>,
        frequency: f64,
        dt: f64,
    ) -> KwaversResult<()> {
        let alpha = self.absorption_coefficient(frequency);

        // Apply exponential decay: exp(-α * c * dt)
        // Assuming c = 1500 m/s for simplicity (should be passed as parameter)
        const SOUND_SPEED: f64 = 1500.0;
        let decay = (-alpha * SOUND_SPEED * dt).exp();

        field.mapv_inplace(|x| x * decay);

        Ok(())
    }
}
