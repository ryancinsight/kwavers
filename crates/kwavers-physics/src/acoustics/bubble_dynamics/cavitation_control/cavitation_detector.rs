//! Cavitation detector facade
//!
//! This module provides a unified interface to the modular detection components
//! in the detection/ subdirectory, maintaining backward compatibility while
//! enforcing proper architectural separation.

// Re-export all detection components
pub use super::detection::{
    BroadbandDetector, CavitationDetectionState, CavitationDetector, CavitationMetrics,
    DetectionMethod, DetectorParameters, SpectralDetector, SubharmonicDetector,
};

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_core::constants::numerical::MHZ_TO_HZ;
    use ndarray::Array1;

    #[test]
    fn test_spectral_detector() {
        let mut detector = SpectralDetector::new(MHZ_TO_HZ, 10.0 * MHZ_TO_HZ);
        let signal = Array1::zeros(1024);
        let metrics = detector.detect(&signal.view());
        assert_eq!(metrics.state, CavitationDetectionState::None);
    }

    #[test]
    fn test_broadband_detector() {
        let mut detector = BroadbandDetector::new(10.0 * MHZ_TO_HZ);
        let signal = Array1::zeros(1024);
        let metrics = detector.detect(&signal.view());
        assert_eq!(metrics.state, CavitationDetectionState::None);
    }

    #[test]
    fn test_subharmonic_detector() {
        let mut detector = SubharmonicDetector::new(MHZ_TO_HZ, 10.0 * MHZ_TO_HZ);
        let signal = Array1::zeros(1024);
        let metrics = detector.detect(&signal.view());
        assert_eq!(metrics.state, CavitationDetectionState::None);
    }
}
