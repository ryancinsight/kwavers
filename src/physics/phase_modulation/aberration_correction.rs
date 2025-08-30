//! Phase Aberration Correction
//!
//! Implements aberration correction methods for transcranial ultrasound.

#[derive(Debug, Debug))]
pub struct AberrationCorrector;
pub enum CorrectionMethod {
    TimeReversal,
    AdaptiveFocusing,
    PhaseConjugation,
}
#[derive(Debug, Debug))]
pub struct TimeReversal;
#[derive(Debug, Debug))]
pub struct AdaptiveFocusing;
#[derive(Debug, Debug))]
pub struct PhaseConjugation;
