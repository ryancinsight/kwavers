//! `UltrafastPlaneWaveConfig` — plane wave imaging configuration.

use aequitas::systems::si::quantities::{Angle, Dimensionless, Frequency, Length, Velocity};
use aequitas::systems::si::units::{Hertz, MeterPerSecond, Radian};
use kwavers_core::constants::fundamental::SOUND_SPEED_TISSUE;

/// Plane wave imaging configuration.
#[derive(Debug, Clone)]
pub struct UltrafastPlaneWaveConfig {
    /// Tilt angles for coherent compounding.
    pub tilt_angles: Vec<Angle<f64>>,
    /// Speed of sound in the configured medium.
    pub sound_speed: Velocity<f64>,
    /// Element positions along the lateral axis.
    pub element_positions: Vec<Length<f64>>,
    /// F-number for apodization (optional).
    pub f_number: Option<Dimensionless<f64>>,
    /// Sampling frequency.
    pub sampling_frequency: Frequency<f64>,
}

impl Default for UltrafastPlaneWaveConfig {
    /// Default: 11 tilted plane waves from −10° to +10° (2° steps),
    /// as in Nouhoum et al. (2021) functional ultrasound protocol.
    fn default() -> Self {
        let tilt_angles: Vec<Angle<f64>> = (-10..=10)
            .step_by(2)
            .map(|a| Angle::from_unit::<Radian>((a as f64).to_radians()))
            .collect();
        Self {
            tilt_angles,
            sound_speed: Velocity::from_unit::<MeterPerSecond>(SOUND_SPEED_TISSUE),
            element_positions: Vec::new(),
            f_number: Some(Dimensionless::from_base(1.5)),
            sampling_frequency: Frequency::from_unit::<Hertz>(40e6),
        }
    }
}
