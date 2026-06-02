//! Skull Attenuation Modeling for Transcranial Ultrasound
//!
//! This module implements comprehensive skull bone attenuation models including
//! frequency-dependent absorption, scattering losses, and dispersion effects
//! critical for accurate transcranial ultrasound simulation and therapy planning.
//!
//! # Mathematical Model
//!
//! Total attenuation coefficient:
//! ```text
//! α_total(f) = α_absorption(f) + α_scattering(f)
//! ```
//!
//! ## References
//! - Pinton, G., et al. (2012). "Attenuation, scattering, and absorption of
//!   ultrasound in the skull bone." J. Acoust. Soc. Am., 131(6), 4694-4706.
//! - White, P. J., et al. (2006). "Use of a theoretical model to assess the
//!   effect of skull curvature on the field of a HIFU transducer."
//! - Fry, F. J., & Barger, J. E. (1978). "Acoustical properties of the human skull."

pub mod field;
pub mod model;
pub mod physics;
pub mod types;

#[cfg(test)]
mod tests;

pub use model::SkullAttenuation;
pub use types::BoneType;
