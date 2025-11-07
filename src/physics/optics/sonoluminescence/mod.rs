//! Sonoluminescence light emission models
//!
//! This module implements the optical physics of sonoluminescence including:
//! - Blackbody radiation from hot bubble interior
//! - Bremsstrahlung from ionized gas
//! - Molecular emission lines
//! - Spectral analysis

pub mod blackbody;
pub mod bremsstrahlung;
pub mod emission;
pub mod spectral;

pub use blackbody::BlackbodyModel;
pub use bremsstrahlung::BremsstrahlungModel;
pub use emission::{
    EmissionParameters, IntegratedSonoluminescence, SonoluminescenceEmission, SpectralField, SpectralStatistics,
};
pub use spectral::{EmissionSpectrum, SpectralAnalyzer};
