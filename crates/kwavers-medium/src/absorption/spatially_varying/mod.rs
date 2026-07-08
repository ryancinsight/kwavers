//! Spatially-varying power law absorption model
//!
//! Implements heterogeneous absorption where both α₀ and γ vary spatially.
//!
//! # Theory
//!
//! ```text
//! α(x, y, z, f) = α₀(x, y, z) · (f / f_ref)^γ(x, y, z)
//! ```
//!
//! # References
//!
//! - Pinton et al. (2009), IEEE Trans. UFFC, 56(3), 474-488.
//! - Treeby & Cox (2010): "Modeling power law absorption and dispersion"

use leto::Array3;

mod computation;
mod construction;
mod mutation;
#[cfg(test)]
mod tests;

/// Spatially-varying power law absorption model
#[derive(Debug, Clone)]
pub struct SpatiallyVaryingAbsorption {
    pub(super) alpha_0_field: Array3<f64>,
    pub(super) gamma_field: Array3<f64>,
    pub(super) f_ref: f64,
    pub(super) dispersion_correction: bool,
    pub(super) temperature_field: Option<Array3<f64>>,
    pub(super) temperature_coefficient: f64,
    pub(super) reference_temperature: f64,
}

/// Statistics of absorption field properties
#[derive(Debug, Clone, Copy)]
pub struct AbsorptionStatistics {
    pub alpha_0_min: f64,
    pub alpha_0_max: f64,
    pub alpha_0_mean: f64,
    pub gamma_min: f64,
    pub gamma_max: f64,
    pub gamma_mean: f64,
}

impl SpatiallyVaryingAbsorption {
    #[must_use]
    pub fn alpha_0_field(&self) -> &Array3<f64> {
        &self.alpha_0_field
    }

    #[must_use]
    pub fn gamma_field(&self) -> &Array3<f64> {
        &self.gamma_field
    }

    pub fn alpha_0_field_mut(&mut self) -> &mut Array3<f64> {
        &mut self.alpha_0_field
    }

    pub fn gamma_field_mut(&mut self) -> &mut Array3<f64> {
        &mut self.gamma_field
    }
}
