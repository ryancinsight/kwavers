//! Configuration for diverging wave (virtual source) imaging.

use aequitas::systems::si::quantities::{Dimensionless, Frequency, Length, Velocity};
use aequitas::systems::si::units::{Hertz, Meter, MeterPerSecond};
use kwavers_core::constants::fundamental::SOUND_SPEED_TISSUE;

/// Diverging wave (virtual source) imaging configuration
#[derive(Debug, Clone)]
pub struct DivergingWaveConfig {
    /// Lateral positions of transducer elements.
    pub element_positions: Vec<Length<f64>>,
    /// Speed of sound in the configured medium.
    pub sound_speed: Velocity<f64>,
    /// Virtual source depth behind the transducer face (positive value).
    ///
    /// A larger F creates a broader diverging wave (wider field of view).
    /// Typical range: 5–20 mm (5–20× element pitch).
    pub virtual_source_depth: Length<f64>,
    /// F-number for apodization (default 1.5)
    ///
    /// Higher F-number → narrower apodization → better side-lobe suppression.
    pub f_number: Dimensionless<f64>,
    /// Sampling frequency used for converting delays to sample indices.
    pub sampling_frequency: Frequency<f64>,
}

impl Default for DivergingWaveConfig {
    fn default() -> Self {
        // 128-element array with 0.3 mm pitch (cardiac imaging, Papadacci et al. 2014)
        let n_elem = 128usize;
        let pitch = 3.0e-4; // 0.3 mm
        let x_start = -(n_elem as f64 - 1.0) / 2.0 * pitch;
        let element_positions: Vec<Length<f64>> = (0..n_elem)
            .map(|i| Length::from_unit::<Meter>((i as f64).mul_add(pitch, x_start)))
            .collect();

        Self {
            element_positions,
            sound_speed: Velocity::from_unit::<MeterPerSecond>(SOUND_SPEED_TISSUE),
            virtual_source_depth: Length::from_unit::<Meter>(0.010),
            f_number: Dimensionless::from_base(1.5),
            sampling_frequency: Frequency::from_unit::<Hertz>(40.0e6),
        }
    }
}
