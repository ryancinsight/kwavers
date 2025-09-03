//! Beamforming configuration

use crate::physics::constants::{SAMPLING_FREQUENCY_DEFAULT, SOUND_SPEED_TISSUE};

/// Configuration for beamforming operations
#[derive(Debug, Clone)]
pub struct BeamformingConfig {
    /// Sound speed in medium (m/s)
    pub sound_speed: f64,
    /// Sampling frequency (Hz)
    pub sampling_frequency: f64,
    /// Reference frequency for array design (Hz)
    pub reference_frequency: f64,
    /// Diagonal loading factor for regularization
    pub diagonal_loading: f64,
    /// Number of snapshots for covariance estimation
    pub num_snapshots: usize,
    /// Spatial smoothing factor
    pub spatial_smoothing: Option<usize>,
}

impl Default for BeamformingConfig {
    fn default() -> Self {
        const REFERENCE_FREQUENCY: f64 = 5e6; // 5 MHz
        const DIAGONAL_LOADING_FACTOR: f64 = 0.01; // 1% diagonal loading
        const DEFAULT_SNAPSHOTS: usize = 100;

        Self {
            sound_speed: SOUND_SPEED_TISSUE,
            sampling_frequency: SAMPLING_FREQUENCY_DEFAULT,
            reference_frequency: REFERENCE_FREQUENCY,
            diagonal_loading: DIAGONAL_LOADING_FACTOR,
            num_snapshots: DEFAULT_SNAPSHOTS,
            spatial_smoothing: None,
        }
    }
}
