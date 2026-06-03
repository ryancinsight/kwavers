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
pub mod histotripsy;
pub mod power_law;
pub mod spatially_varying;
pub mod stokes;
pub mod tissue;

pub use dispersion::{AbsorptionDispersionCorrection, DispersionModel};
pub use fractional::{FractionalDerivative, FractionalLaplacian};
pub use histotripsy::{
    histotripsy_tissue_properties, histotripsy_tissue_properties_by_name,
    HistotripsyTissueProperties,
};
pub use power_law::{PowerLawAbsorption, PowerLawModel};
pub use spatially_varying::{AbsorptionStatistics, SpatiallyVaryingAbsorption};
pub use stokes::{StokesAbsorption, StokesParameters};
pub use tissue::{AbsorptionTissueType, TissueAbsorption, TISSUE_PROPERTIES};

use kwavers_core::error::KwaversResult;
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
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn absorption_coefficient(&self, frequency: f64) -> f64 {
        match &self.model {
            AbsorptionModel::PowerLaw(m) => m.absorption_at_frequency(frequency),
            AbsorptionModel::Tissue(m) => m.absorption_at_frequency(frequency),
            AbsorptionModel::Stokes(m) => m.absorption_at_frequency(frequency),
            AbsorptionModel::Fractional(m) => m.absorption_at_frequency(frequency),
        }
    }

    /// Apply absorption to a field over one time step.
    ///
    /// The exponential decay factor `exp(−α(f)·c·Δt)` is the analytic
    /// plane-wave attenuation rate for one step of duration `dt` along
    /// a path of length `c·dt`.  The sound speed `c` is the local
    /// (or representative) wave speed in the medium and must be passed
    /// by the caller — prior to 2026-05-21 this method silently used
    /// `SOUND_SPEED_WATER_SIM` (1500 m/s) regardless of medium, which
    /// produced wrong attenuation for soft tissue (1540 m/s, ~3 % low),
    /// bone (3500 m/s, ~57 % low), or any non-water material.
    ///
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    pub fn apply_absorption(
        &self,
        field: &mut Array3<f64>,
        frequency: f64,
        dt: f64,
        sound_speed: f64,
    ) -> KwaversResult<()> {
        let alpha = self.absorption_coefficient(frequency);
        let decay = (-alpha * sound_speed * dt).exp();
        field.par_mapv_inplace(|x| x * decay);
        Ok(())
    }
}
