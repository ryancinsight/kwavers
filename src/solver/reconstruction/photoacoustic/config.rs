//! Configuration for photoacoustic reconstruction
//!
//! This module defines the configuration structures for photoacoustic
//! reconstruction algorithms.

use super::algorithms::PhotoacousticAlgorithm;

/// Configuration for photoacoustic reconstruction
#[derive(Debug, Clone))]
pub struct PhotoacousticConfig {
    /// Reconstruction algorithm to use
    pub algorithm: PhotoacousticAlgorithm,
    /// Sensor positions [x, y, z] in meters
    pub sensor_positions: Vec<[f64; 3]>,
    /// Grid size for reconstruction [nx, ny, nz]
    pub grid_size: [usize; 3],
    /// Sound speed in medium (m/s)
    pub sound_speed: f64,
    /// Sampling frequency (Hz)
    pub sampling_frequency: f64,
    /// Apply envelope detection
    pub envelope_detection: bool,
    /// Bandpass filter parameters [low_freq, high_freq] in Hz
    pub bandpass_filter: Option<[f64; 2]>,
    /// Regularization parameter for iterative methods
    pub regularization_parameter: f64,
}
