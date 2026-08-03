//! Beamforming configuration (core, unified)

use aequitas::systems::si::quantities::{Frequency, Velocity};
use aequitas::systems::si::units::{Hertz, MeterPerSecond};
use kwavers_core::constants::numerical::MHZ_TO_HZ;
use kwavers_core::constants::{SAMPLING_FREQUENCY_DEFAULT, SOUND_SPEED_TISSUE};

/// Core configuration for beamforming operations across array-processing consumers.
///
/// This struct is the single source of truth for physical and numerical
/// parameters used by array beamforming. Downstream modules (e.g., PAM and
/// localization) consume this core type directly.
///
#[derive(Debug, Clone)]
pub struct BeamformingCoreConfig {
    /// Sound speed in the medium.
    pub sound_speed: Velocity<f64>,
    /// Sampling frequency.
    pub sampling_frequency: Frequency<f64>,
    /// Reference frequency for array design.
    pub reference_frequency: Frequency<f64>,
    /// Diagonal loading factor for regularization
    pub diagonal_loading: f64,
    /// Number of snapshots for covariance estimation
    pub num_snapshots: usize,
    /// Spatial smoothing factor
    pub spatial_smoothing: Option<usize>,
}

impl Default for BeamformingCoreConfig {
    fn default() -> Self {
        let reference_frequency = Frequency::from_unit::<Hertz>(5.0 * MHZ_TO_HZ);
        const DIAGONAL_LOADING_FACTOR: f64 = 0.01; // 1% diagonal loading
        const DEFAULT_SNAPSHOTS: usize = 100;

        Self {
            sound_speed: Velocity::from_unit::<MeterPerSecond>(SOUND_SPEED_TISSUE),
            sampling_frequency: Frequency::from_unit::<Hertz>(SAMPLING_FREQUENCY_DEFAULT),
            reference_frequency,
            diagonal_loading: DIAGONAL_LOADING_FACTOR,
            num_snapshots: DEFAULT_SNAPSHOTS,
            spatial_smoothing: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_physical_metrics_preserve_si_values() {
        let config = BeamformingCoreConfig::default();

        assert_eq!(
            config.sound_speed.in_unit::<MeterPerSecond>(),
            SOUND_SPEED_TISSUE
        );
        assert_eq!(
            config.sampling_frequency.in_unit::<Hertz>(),
            SAMPLING_FREQUENCY_DEFAULT
        );
        assert_eq!(
            config.reference_frequency.in_unit::<Hertz>(),
            5.0 * MHZ_TO_HZ
        );
    }
}
