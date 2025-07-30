//! Sonoluminescence light emission models
//!
//! This module implements the optical physics of sonoluminescence including:
//! - Blackbody radiation from hot bubble interior
//! - Bremsstrahlung from ionized gas
//! - Molecular emission lines
//! - Spectral analysis

pub mod blackbody;
pub mod bremsstrahlung;
pub mod spectral;
pub mod emission;

pub use blackbody::BlackbodyModel;
pub use bremsstrahlung::BremsstrahlungModel;
pub use spectral::{SpectralAnalyzer, EmissionSpectrum};
pub use emission::{SonoluminescenceEmission, EmissionParameters};