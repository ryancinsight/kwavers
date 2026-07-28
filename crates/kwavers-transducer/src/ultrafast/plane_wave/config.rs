//! `UltrafastPlaneWaveConfig` — plane wave imaging configuration.

use aequitas::systems::si::quantities::{Angle, Frequency, Length, Velocity};
use kwavers_core::constants::fundamental::SOUND_SPEED_TISSUE;

/// Plane wave imaging configuration.
#[derive(Debug, Clone)]
pub struct UltrafastPlaneWaveConfig {
    /// Tilt angles for coherent compounding (radians).
    pub tilt_angles: Vec<Angle>,
    /// Speed of sound.
    pub sound_speed: Velocity,
    /// Element positions on the x axis.
    pub element_positions: Vec<Length>,
    /// F-number for apodization (optional).
    pub f_number: Option<f64>,
    /// Sampling frequency.
    pub sampling_frequency: Frequency,
}

impl Default for UltrafastPlaneWaveConfig {
    /// Default: 11 tilted plane waves from −10° to +10° (2° steps),
    /// as in Nouhoum et al. (2021) functional ultrasound protocol.
    fn default() -> Self {
        let tilt_angles: Vec<Angle> = (-10..=10)
            .step_by(2)
            .map(|a| Angle::from_base((a as f64).to_radians()))
            .collect();
        Self {
            tilt_angles,
            sound_speed: Velocity::from_base(SOUND_SPEED_TISSUE),
            element_positions: Vec::new(),
            f_number: Some(1.5),
            sampling_frequency: Frequency::from_base(40e6),
        }
    }
}
