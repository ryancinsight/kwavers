//! Harmonic Detection and Analysis for Nonlinear SWE
//!
//! Implements multi-frequency displacement tracking and harmonic analysis
//! for nonlinear shear wave elastography.

pub mod config;
pub mod detector;
pub mod spectral;
pub mod types;

#[cfg(test)]
mod tests;

pub use config::HarmonicDetectionConfig;
pub use detector::HarmonicDetector;
pub use types::HarmonicDisplacementField;
