//! Cavitation detector facade
//!
//! This module provides a unified interface to the modular detection components
//! in the detection/ subdirectory, maintaining backward compatibility while
//! enforcing proper architectural separation.

// Re-export all detection components
pub use super::detection::{
    BroadbandDetector, CavitationDetector, CavitationMetrics, CavitationState, DetectionMethod,
    DetectorParameters, SpectralDetector, SubharmonicDetector,
};

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_spectral_detector() {
        let mut detector = SpectralDetector::new(1e6, 10e6);
        let signal = Array1::zeros(1024);
        let metrics = detector.detect(&signal.view());
        assert_eq!(metrics.state, CavitationState::None);
    }

    #[test]
    fn test_broadband_detector() {
        let mut detector = BroadbandDetector::new(10e6);
        let signal = Array1::zeros(1024);
        let metrics = detector.detect(&signal.view());
        assert_eq!(metrics.state, CavitationState::None);
    }

    #[test]
    fn test_subharmonic_detector() {
        let mut detector = SubharmonicDetector::new(1e6, 10e6);
        let signal = Array1::zeros(1024);
        let metrics = detector.detect(&signal.view());
        assert_eq!(metrics.state, CavitationState::None);
    }
}
