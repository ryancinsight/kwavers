//! Phase Aberration Correction
//!
//! Implements aberration correction methods for transcranial ultrasound.

pub struct AberrationCorrector;
pub enum CorrectionMethod {
    TimeReversal,
    AdaptiveFocusing,
    PhaseConjugation,
}
pub struct TimeReversal;
pub struct AdaptiveFocusing;
pub struct PhaseConjugation;
