//! Cavitation detection and monitoring for therapy
//!
//! Provides cavitation detection methods for therapeutic ultrasound applications.

use ndarray::Array3;

/// Physical constants
const WATER_SURFACE_TENSION: f64 = 0.0728; // N/m at 20°C
const WATER_VAPOR_PRESSURE: f64 = 2.34e3; // Pa at 20°C
const ATMOSPHERIC_PRESSURE: f64 = 101325.0; // Pa

/// Cavitation detector for therapy
pub struct TherapyCavitationDetector {
    /// Frequency [Hz]
    frequency: f64,
    /// Blake threshold pressure [Pa]
    pub blake_threshold: f64,
    /// Detection method
    method: CavitationDetectionMethod,
}

/// Cavitation detection methods
#[derive(Debug, Clone, Copy)]
pub enum CavitationDetectionMethod {
    /// Pressure threshold based
    PressureThreshold,
    /// Spectral analysis based
    Spectral,
    /// Combined methods
    Combined,
}

impl TherapyCavitationDetector {
    /// Create a new cavitation detector
    pub fn new(frequency: f64, peak_negative_pressure: f64) -> Self {
        // Blake threshold calculation
        // P_Blake = P0 + P_v - 2σ/R0
        // Simplified for micron-sized nuclei
        let r0 = 1e-6; // 1 μm nucleus
        let blake_threshold =
            ATMOSPHERIC_PRESSURE + WATER_VAPOR_PRESSURE - 2.0 * WATER_SURFACE_TENSION / r0;

        Self {
            frequency,
            blake_threshold,
            method: CavitationDetectionMethod::PressureThreshold,
        }
    }

    /// Detect cavitation in pressure field
    pub fn detect(&self, pressure: &Array3<f64>) -> Array3<bool> {
        let mut cavitation = Array3::from_elem(pressure.dim(), false);

        match self.method {
            CavitationDetectionMethod::PressureThreshold => {
                self.detect_by_threshold(pressure, &mut cavitation);
            }
            CavitationDetectionMethod::Spectral => {
                self.detect_by_spectral(pressure, &mut cavitation);
            }
            CavitationDetectionMethod::Combined => {
                self.detect_by_threshold(pressure, &mut cavitation);
                // Additional spectral detection could be added
            }
        }

        cavitation
    }

    /// Detect cavitation by pressure threshold
    fn detect_by_threshold(&self, pressure: &Array3<f64>, cavitation: &mut Array3<bool>) {
        cavitation
            .iter_mut()
            .zip(pressure.iter())
            .for_each(|(cav, &p)| {
                // Cavitation occurs when negative pressure exceeds Blake threshold
                *cav = -p > self.blake_threshold;
            });
    }

    /// Detect cavitation by spectral analysis (placeholder)
    fn detect_by_spectral(&self, _pressure: &Array3<f64>, cavitation: &mut Array3<bool>) {
        // Spectral detection would analyze frequency content
        // Looking for subharmonics, ultraharmonics, and broadband noise
        // This is a simplified placeholder
        cavitation.fill(false);
    }

    /// Calculate cavitation index
    pub fn cavitation_index(&self, peak_negative_pressure: f64) -> f64 {
        // CI = |P_neg| / P_Blake
        peak_negative_pressure.abs() / self.blake_threshold
    }

    /// Estimate cavitation probability
    pub fn cavitation_probability(&self, peak_negative_pressure: f64) -> f64 {
        let ci = self.cavitation_index(peak_negative_pressure);

        // Sigmoid function for probability
        // P(cav) = 1 / (1 + exp(-k*(CI - 1)))
        let k = 5.0; // Steepness parameter
        1.0 / (1.0 + ((-k * (ci - 1.0)).exp()))
    }

    /// Check if conditions are safe for stable cavitation
    pub fn is_stable_cavitation(&self, peak_negative_pressure: f64) -> bool {
        let ci = self.cavitation_index(peak_negative_pressure);
        // Stable cavitation typically occurs for 0.5 < CI < 1.0
        ci > 0.5 && ci < 1.0
    }

    /// Check if conditions lead to inertial cavitation
    pub fn is_inertial_cavitation(&self, peak_negative_pressure: f64) -> bool {
        let ci = self.cavitation_index(peak_negative_pressure);
        // Inertial cavitation typically occurs for CI > 1.0
        ci > 1.0
    }
}
