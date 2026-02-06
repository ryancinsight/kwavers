//! Beamforming Result Types

use ndarray::{Array1, Array2, Array3};

/// Beamformed signal result
#[derive(Debug, Clone)]
pub struct BeamformingResult {
    /// Beamformed output (time or frequency domain)
    pub beamformed_signal: Array3<f64>,

    /// Beam pattern (spatial directivity function)
    pub beam_pattern: BeamPattern,

    /// Confidence or SNR estimate
    pub confidence: Array1<f64>,

    /// Processing metadata
    pub metadata: std::collections::HashMap<String, String>,
}

impl BeamformingResult {
    /// Create new beamforming result
    pub fn new(
        beamformed_signal: Array3<f64>,
        beam_pattern: BeamPattern,
        confidence: Array1<f64>,
    ) -> Self {
        Self {
            beamformed_signal,
            beam_pattern,
            confidence,
            metadata: Default::default(),
        }
    }
}

/// Beam pattern (normalized directivity function)
///
/// Represents the spatial response of the beamformer as a function of direction.
#[derive(Debug, Clone)]
pub struct BeamPattern {
    /// Beam response in angle coordinates (azimuth × elevation)
    pub response: Array2<f64>,

    /// Azimuth angles [radians]
    pub azimuth: Array1<f64>,

    /// Elevation angles [radians]
    pub elevation: Array1<f64>,

    /// Beamwidth (-3dB) [radians]
    pub beamwidth_3db: f64,

    /// Main lobe gain [dB]
    pub main_lobe_level: f64,

    /// Side lobe level [dB]
    pub side_lobe_level: f64,
}

impl BeamPattern {
    /// Create new beam pattern
    pub fn new(
        response: Array2<f64>,
        azimuth: Array1<f64>,
        elevation: Array1<f64>,
        beamwidth_3db: f64,
    ) -> Self {
        let main_lobe_level = response
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .copied()
            .unwrap_or(0.0)
            .log10()
            * 20.0;

        let side_lobe_level = response
            .iter()
            .filter(|&&v| v < 0.5) // Below -6dB
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .copied()
            .unwrap_or(0.01)
            .log10()
            * 20.0;

        Self {
            response,
            azimuth,
            elevation,
            beamwidth_3db,
            main_lobe_level,
            side_lobe_level,
        }
    }

    /// Get directivity index [dB]
    ///
    /// Measures how directional the beam is compared to omnidirectional
    pub fn directivity_index(&self) -> f64 {
        // Simplified: depends on main lobe vs. side lobes
        self.main_lobe_level - self.side_lobe_level
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn test_beamforming_result_creation() {
        let signal = Array3::zeros((10, 10, 10));
        let beam_pattern = BeamPattern::new(
            Array2::zeros((64, 64)),
            Array1::zeros(64),
            Array1::zeros(64),
            0.1,
        );
        let confidence = Array1::ones(10);

        let result = BeamformingResult::new(signal, beam_pattern, confidence);
        assert_eq!(result.beamformed_signal.dim(), (10, 10, 10));
    }

    #[test]
    fn test_beam_pattern_creation() {
        let response = Array::from_elem((64, 64), 0.5);
        let azimuth = Array1::linspace(0.0, 2.0 * std::f64::consts::PI, 64);
        let elevation =
            Array1::linspace(-std::f64::consts::PI / 2.0, std::f64::consts::PI / 2.0, 64);

        let pattern = BeamPattern::new(response, azimuth, elevation, 0.1);
        assert!(pattern.main_lobe_level < 0.0); // Should be in dB
    }
}
