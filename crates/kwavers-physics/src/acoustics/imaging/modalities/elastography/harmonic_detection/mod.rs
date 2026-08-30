//! Harmonic Detection and Analysis for Nonlinear SWE
//!
//! Implements multi-frequency displacement tracking and whole-record harmonic
//! analysis for nonlinear shear wave elastography. Each spatial trace receives
//! one symmetric Hann window and one FFT; callers own any record segmentation,
//! overlap policy, SNR filtering, or phase unwrapping required by their domain.

pub mod config;
pub mod detector;
pub mod spectral;
pub mod types;

#[cfg(test)]
mod tests;

pub use config::HarmonicDetectionConfig;
pub use detector::HarmonicDetector;
pub use types::HarmonicDisplacementField;
