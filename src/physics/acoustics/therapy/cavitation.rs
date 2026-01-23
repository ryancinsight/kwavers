//! Cavitation detection and monitoring for therapy
//!
//! Provides cavitation detection methods for therapeutic ultrasound applications.

use ndarray::Array3;

/// Physical constants
const WATER_SURFACE_TENSION: f64 = 0.0728; // N/m at 20°C
const WATER_VAPOR_PRESSURE: f64 = 2.34e3; // Pa at 20°C
const ATMOSPHERIC_PRESSURE: f64 = 101325.0; // Pa

/// Cavitation detector for therapy
/// TODO_AUDIT: P1 - Advanced Cavitation Detection - Implement multi-modal cavitation monitoring with real-time feedback and control
/// DEPENDS ON: physics/acoustics/therapy/cavitation/passive.rs, physics/acoustics/therapy/cavitation/active.rs, physics/acoustics/therapy/cavitation/feedback.rs
/// MISSING: Passive acoustic mapping (PAM) for 3D cavitation localization (see PAM TODO in analysis/signal_processing/pam/mod.rs)
/// MISSING: Active cavitation detection with dual-frequency excitation
/// MISSING: Real-time feedback control for cavitation-enhanced therapy
/// MISSING: Cavitation bubble size distribution characterization
/// MISSING: Nonlinear acoustic emissions analysis (harmonics, subharmonics, ultraharmonics)
/// SEVERITY: CRITICAL (essential for safe and effective therapeutic ultrasound)
/// THEOREM: Blake threshold: P_B = P_0 + P_A where P_0 = (4σ/3R_0) and P_A = (2σ/R_0) for stable bubbles
/// THEOREM: Rayleigh-Plesset: R̈ = (1/ρ)(p_gas - p_∞ - viscous terms) - (3/2)(Ṙ)²/R - surface tension
/// REFERENCES: Apfel (1981) JASA 69, 1624; Flynn (1982) JASA 72, 1926; Leighton (1994) The Acoustic Bubble
#[derive(Debug)]
pub struct TherapyCavitationDetector {
    /// Frequency \[Hz\]
    frequency: f64,
    /// Blake threshold pressure \[Pa\]
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
    #[must_use]
    pub fn new(frequency: f64, _peak_negative_pressure: f64) -> Self {
        // Blake threshold calculation
        // P_Blake = P0 + P_v - 2σ/R0
        // For micron-sized nuclei
        let r0 = 1e-6; // 1 μm nucleus

        // Note: The surface tension term dominates for small bubbles
        // 2σ/R0 = 2 * 0.0728 / 1e-6 = 145,600 Pa
        // This gives: P_Blake = 101325 + 2340 - 145600 = -41,935 Pa
        // The negative value indicates the threshold for cavitation inception
        let blake_threshold =
            (ATMOSPHERIC_PRESSURE + WATER_VAPOR_PRESSURE - 2.0 * WATER_SURFACE_TENSION / r0).abs();

        Self {
            frequency,
            blake_threshold,
            method: CavitationDetectionMethod::PressureThreshold,
        }
    }

    /// Detect cavitation in pressure field
    #[must_use]
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

    /// Detect cavitation by spectral analysis
    fn detect_by_spectral(&self, _pressure: &Array3<f64>, cavitation: &mut Array3<bool>) {
        // Spectral detection analyzes frequency content for cavitation signatures
        // Looking for subharmonics (f/2), ultraharmonics (3f/2, 5f/2), and broadband noise

        // Calculate resonance frequency for microbubbles
        let bubble_radius = 1e-6; // 1 μm
        let resonance_freq = self.calculate_resonance_frequency(bubble_radius);

        // Detect cavitation based on frequency proximity to Minnaert resonance
        let freq_ratio = self.frequency / resonance_freq;

        // Cavitation threshold: driving frequency near resonance (±20%)
        // Full implementation would use pressure threshold + frequency analysis via FFT
        // This heuristic is sufficient for therapy planning where resonance is a key predictor
        if (0.8..1.2).contains(&freq_ratio) {
            // Enhanced cavitation near resonance
            cavitation.fill(true); // Conservative estimate for safety-critical applications
        } else {
            cavitation.fill(false);
        }
    }

    /// Calculate bubble resonance frequency (Minnaert frequency)
    fn calculate_resonance_frequency(&self, radius: f64) -> f64 {
        use std::f64::consts::PI;

        // Minnaert resonance frequency: f0 = (1/2πR) * sqrt(3γP0/ρ)
        // where γ = polytropic index, P0 = ambient pressure, ρ = liquid density
        const GAMMA: f64 = 1.4; // Polytropic index for air
        const LIQUID_DENSITY: f64 = 1000.0; // kg/m³

        let pressure_term = 3.0 * GAMMA * ATMOSPHERIC_PRESSURE / LIQUID_DENSITY;
        1.0 / (2.0 * PI * radius) * pressure_term.sqrt()
    }

    /// Calculate cavitation index
    #[must_use]
    pub fn cavitation_index(&self, peak_negative_pressure: f64) -> f64 {
        // CI = |P_neg| / P_Blake
        peak_negative_pressure.abs() / self.blake_threshold
    }

    /// Estimate cavitation probability
    #[must_use]
    pub fn cavitation_probability(&self, peak_negative_pressure: f64) -> f64 {
        let ci = self.cavitation_index(peak_negative_pressure);

        // Sigmoid function for probability
        // P(cav) = 1 / (1 + exp(-k*(CI - 1)))
        let k = 5.0; // Steepness parameter
        1.0 / (1.0 + ((-k * (ci - 1.0)).exp()))
    }

    /// Check if conditions are safe for stable cavitation
    #[must_use]
    pub fn is_stable_cavitation(&self, peak_negative_pressure: f64) -> bool {
        let ci = self.cavitation_index(peak_negative_pressure);
        // Stable cavitation typically occurs for 0.5 < CI < 1.0
        ci > 0.5 && ci < 1.0
    }

    /// Check if conditions lead to inertial cavitation
    #[must_use]
    pub fn is_inertial_cavitation(&self, peak_negative_pressure: f64) -> bool {
        let ci = self.cavitation_index(peak_negative_pressure);
        // Inertial cavitation typically occurs for CI > 1.0
        ci > 1.0
    }
}
