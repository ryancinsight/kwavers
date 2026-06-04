//! `UltrafastPlaneWaveConfig` — plane wave imaging configuration.

use kwavers_core::constants::fundamental::SOUND_SPEED_TISSUE;

/// Plane wave imaging configuration.
#[derive(Debug, Clone)]
pub struct UltrafastPlaneWaveConfig {
    /// Tilt angles for coherent compounding (radians).
    pub tilt_angles: Vec<f64>,
    /// Speed of sound (m/s).
    pub sound_speed: f64,
    /// Element positions (x coordinates, meters).
    pub element_positions: Vec<f64>,
    /// F-number for apodization (optional).
    pub f_number: Option<f64>,
    /// Sampling frequency (Hz).
    pub sampling_frequency: f64,
}

impl Default for UltrafastPlaneWaveConfig {
    /// Default: 11 tilted plane waves from −10° to +10° (2° steps),
    /// as in Nouhoum et al. (2021) functional ultrasound protocol.
    fn default() -> Self {
        let tilt_angles: Vec<f64> = (-10..=10)
            .step_by(2)
            .map(|a| (a as f64).to_radians())
            .collect();
        Self {
            tilt_angles,
            sound_speed: SOUND_SPEED_TISSUE,
            element_positions: Vec::new(),
            f_number: Some(1.5),
            sampling_frequency: 40e6,
        }
    }
}
