//! Configuration for diverging wave (virtual source) imaging.

use crate::core::constants::fundamental::SOUND_SPEED_TISSUE;

/// Diverging wave (virtual source) imaging configuration
#[derive(Debug, Clone)]
pub struct DivergingWaveConfig {
    /// Lateral positions of transducer elements (m)
    pub element_positions: Vec<f64>,
    /// Speed of sound in medium (m/s)
    pub sound_speed: f64,
    /// Virtual source depth behind the transducer face (m, positive value)
    ///
    /// A larger F creates a broader diverging wave (wider field of view).
    /// Typical range: 5–20 mm (5–20× element pitch).
    pub virtual_source_depth: f64,
    /// F-number for apodization (default 1.5)
    ///
    /// Higher F-number → narrower apodization → better side-lobe suppression.
    pub f_number: f64,
    /// Sampling frequency (Hz), used for converting delays to sample indices
    pub sampling_frequency: f64,
}

impl Default for DivergingWaveConfig {
    fn default() -> Self {
        // 128-element array with 0.3 mm pitch (cardiac imaging, Papadacci et al. 2014)
        let n_elem = 128usize;
        let pitch = 3.0e-4; // 0.3 mm
        let x_start = -(n_elem as f64 - 1.0) / 2.0 * pitch;
        let element_positions: Vec<f64> = (0..n_elem)
            .map(|i| (i as f64).mul_add(pitch, x_start))
            .collect();

        Self {
            element_positions,
            sound_speed: SOUND_SPEED_TISSUE, // Soft tissue
            virtual_source_depth: 0.010,     // 10 mm behind transducer
            f_number: 1.5,
            sampling_frequency: 40.0e6, // 40 MHz
        }
    }
}
