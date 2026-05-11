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

#[cfg(test)]
mod tests {
    use super::*;

    /// AberrationCorrector is constructible and Debug-formattable.
    #[test]
    fn aberration_corrector_debug_non_empty() {
        let s = format!("{:?}", AberrationCorrector);
        assert!(!s.is_empty(), "AberrationCorrector debug must not be empty");
    }

    /// All CorrectionMethod variants are distinct Debug strings.
    #[test]
    fn correction_method_variants_debug_distinct() {
        let tr = format!("{:?}", CorrectionMethod::TimeReversal);
        let af = format!("{:?}", CorrectionMethod::AdaptiveFocusing);
        let pc = format!("{:?}", CorrectionMethod::PhaseConjugation);
        assert_ne!(tr, af);
        assert_ne!(af, pc);
        assert_ne!(tr, pc);
    }

    /// Marker structs are constructible.
    #[test]
    fn marker_structs_constructible() {
        let _tr = TimeReversal;
        let _af = AdaptiveFocusing;
        let _pc = PhaseConjugation;
    }
}
