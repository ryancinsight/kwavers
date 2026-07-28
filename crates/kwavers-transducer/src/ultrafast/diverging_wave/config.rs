//! Configuration for diverging wave (virtual source) imaging.

use aequitas::systems::si::quantities::{Frequency, Length, Velocity};
use kwavers_core::constants::fundamental::SOUND_SPEED_TISSUE;

/// Diverging wave (virtual source) imaging configuration
#[derive(Debug, Clone)]
pub struct DivergingWaveConfig {
    /// Lateral positions of transducer elements.
    pub element_positions: Vec<Length>,
    /// Speed of sound in the medium.
    pub sound_speed: Velocity,
    /// Virtual source depth behind the transducer face (positive value).
    ///
    /// A larger F creates a broader diverging wave (wider field of view).
    /// Typical range: 5–20 mm (5–20× element pitch).
    pub virtual_source_depth: Length,
    /// F-number for apodization (default 1.5)
    ///
    /// Higher F-number → narrower apodization → better side-lobe suppression.
    pub f_number: f64,
    /// Sampling frequency, used for converting delays to sample indices.
    pub sampling_frequency: Frequency,
}

impl Default for DivergingWaveConfig {
    fn default() -> Self {
        // 128-element array with 0.3 mm pitch (cardiac imaging, Papadacci et al. 2014)
        let n_elem = 128usize;
        let pitch = 3.0e-4; // 0.3 mm
        let x_start = -(n_elem as f64 - 1.0) / 2.0 * pitch;
        let element_positions: Vec<Length> = (0..n_elem)
            .map(|i| Length::from_base((i as f64).mul_add(pitch, x_start)))
            .collect();

        Self {
            element_positions,
            sound_speed: Velocity::from_base(SOUND_SPEED_TISSUE),
            virtual_source_depth: Length::from_base(0.010),
            f_number: 1.5,
            sampling_frequency: Frequency::from_base(40.0e6),
        }
    }
}
