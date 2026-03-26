//! Aberration Correction for Transcranial Ultrasound
//!
//! Implements phase correction algorithms to compensate for skull-induced
//! phase aberrations in transcranial focused ultrasound.

pub mod adaptive;
pub mod phase_correction;
pub mod time_reversal;
pub mod validation;

#[cfg(test)]
mod tests;

pub use phase_correction::{PhaseCorrection, TranscranialAberrationCorrection};
pub use validation::CorrectionValidation;
