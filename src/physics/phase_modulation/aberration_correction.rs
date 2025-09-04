//! Phase Aberration Correction
//!
//! Implements aberration correction methods for transcranial ultrasound.

#[derive(Debug)]
pub struct AberrationCorrector;
#[derive(Debug)]
pub enum CorrectionMethod {
    TimeReversal,
    AdaptiveFocusing,
    PhaseConjugation,
}
#[derive(Debug)]
pub struct TimeReversal;
#[derive(Debug)]
pub struct AdaptiveFocusing;
#[derive(Debug)]
pub struct PhaseConjugation;
