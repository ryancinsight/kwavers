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
pub mod relaxation_fit;
pub mod spatially_varying;
pub mod stokes;
pub mod tissue;

use aequitas::systems::si::quantities::{Frequency, ReciprocalLength, Time, Velocity};
pub use dispersion::AbsorptionDispersionCorrection;
pub use fractional::{FractionalAbsorptionStep, FractionalDerivative, FractionalLaplacian};
pub use histotripsy::{
    histotripsy_tissue_properties, histotripsy_tissue_properties_by_name,
    HistotripsyTissueProperties,
};
pub use power_law::{PowerLawAbsorption, PowerLawModel};
pub use relaxation_fit::{
    fit_power_law, fit_power_law_fields, FitBand, PowerLawTarget, RelaxationFieldFit,
    RelaxationSpectrumFit, RelaxationTimePlacement,
};
pub use spatially_varying::{AbsorptionStatistics, SpatiallyVaryingAbsorption};
pub use stokes::{StokesAbsorption, StokesParameters};
pub use tissue::{
    tissue_thermal_properties, AbsorptionTissueType, TissueAbsorption, TISSUE_PROPERTIES,
};

use kwavers_core::error::KwaversResult;
use leto::Array3;

use kwavers_core::traversal::zip_mut;

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

    /// Evaluate amplitude loss at the supplied physical frequency.
    ///
    /// # Errors
    /// Returns an error if the frequency or evaluated attenuation is invalid.
    pub fn absorption_coefficient(&self, frequency: Frequency) -> KwaversResult<ReciprocalLength> {
        let hertz = frequency.into_base();
        if !hertz.is_finite() || hertz < 0.0 {
            return Err(kwavers_core::error::KwaversError::InvalidInput(
                "Frequency must be finite and nonnegative".into(),
            ));
        }
        let loss = match &self.model {
            AbsorptionModel::PowerLaw(model) => return model.absorption_at_frequency(frequency),
            AbsorptionModel::Tissue(model) => model.absorption_at_frequency(hertz),
            AbsorptionModel::Stokes(model) => model.absorption_at_frequency(hertz),
            AbsorptionModel::Fractional(model) => return model.absorption_at_frequency(frequency),
        };
        if !loss.is_finite() || loss < 0.0 {
            return Err(kwavers_core::error::KwaversError::InvalidInput(
                "Invalid evaluated amplitude attenuation".into(),
            ));
        }
        Ok(ReciprocalLength::from_base(loss))
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
        frequency: Frequency,
        dt: Time,
        sound_speed: Velocity,
    ) -> KwaversResult<()> {
        if !dt.as_base().is_finite()
            || dt.into_base() < 0.0
            || !sound_speed.as_base().is_finite()
            || sound_speed.into_base() <= 0.0
        {
            return Err(kwavers_core::error::KwaversError::InvalidInput(
                "Time must be finite and nonnegative and sound speed must be finite and positive"
                    .into(),
            ));
        }
        let alpha = self.absorption_coefficient(frequency)?;
        let distance = sound_speed * dt;
        if !distance.as_base().is_finite() {
            return Err(kwavers_core::error::KwaversError::InvalidInput(
                "Propagation distance exceeds scalar range".into(),
            ));
        }
        let decay = (-(alpha * distance).into_base()).exp();
        zip_mut(field.view_mut(), (), |x, ()| *x *= decay);
        Ok(())
    }
}
