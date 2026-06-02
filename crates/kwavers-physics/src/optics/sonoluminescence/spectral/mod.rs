//! Spectral analysis for sonoluminescence
//!
//! Tools for analyzing emission spectra and extracting physical parameters

pub mod analyzer;
pub mod range;
pub mod spectrum;

#[cfg(test)]
mod tests;

pub use analyzer::SpectralAnalyzer;
pub use range::SpectralRange;
pub use spectrum::EmissionSpectrum;
