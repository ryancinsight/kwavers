//! Reconstruction configuration types and traits

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use ndarray::{Array2, Array3};
use serde::{Deserialize, Serialize};

/// Reconstruction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReconstructionConfig {
    /// Speed of sound in medium (m/s)
    pub sound_speed: f64,
    /// Sampling frequency (Hz)
    pub sampling_frequency: f64,
    /// Reconstruction algorithm
    pub algorithm: ReconstructionAlgorithm,
    /// Filter type for reconstruction
    pub filter: ReconstructionFilterType,
    /// Interpolation method
    pub interpolation: ReconstructionInterpolationMethod,
}

impl Default for ReconstructionConfig {
    fn default() -> Self {
        use crate::core::constants::fundamental::SOUND_SPEED_TISSUE;
        Self {
            sound_speed: SOUND_SPEED_TISSUE, // Typical speed of sound in water/soft tissue
            sampling_frequency: 40e6, // 40 MHz typical for ultrasound imaging
            algorithm: ReconstructionAlgorithm::BackProjection,
            filter: ReconstructionFilterType::Hamming,
            interpolation: ReconstructionInterpolationMethod::Linear,
        }
    }
}

/// Reconstruction algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReconstructionAlgorithm {
    /// Universal back-projection algorithm
    UniversalBackProjection,
    /// Back-projection algorithm
    BackProjection,
    /// Filtered back-projection
    FilteredBackProjection,
    /// Time reversal
    PhotoacousticTimeReversal,
    /// Fourier-domain reconstruction
    FourierDomain,
    /// Iterative reconstruction
    Iterative { iterations: usize },
    /// Full Waveform Inversion for seismic imaging
    FullWaveformInversion,
    /// Reverse Time Migration for seismic imaging
    ReverseTimeMigration,
}

/// Filter types for reconstruction
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ReconstructionFilterType {
    /// No filtering
    None,
    /// Ram-Lak filter
    RamLak,
    /// Shepp-Logan filter
    SheppLogan,
    /// Cosine filter
    Cosine,
    /// Hamming window
    Hamming,
    /// Hann window
    Hann,
}

/// Interpolation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReconstructionInterpolationMethod {
    /// Nearest neighbor
    NearestNeighbor,
    /// Linear interpolation
    Linear,
    /// Cubic interpolation
    Cubic,
    /// Sinc interpolation
    Sinc,
}

/// Base trait for all reconstruction methods
pub trait Reconstructor {
    /// Perform reconstruction from sensor data
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn reconstruct(
        &self,
        sensor_data: &Array2<f64>, // [sensors x time_steps]
        sensor_positions: &[[f64; 3]],
        grid: &Grid,
        config: &ReconstructionConfig,
    ) -> KwaversResult<Array3<f64>>;

    /// Get reconstruction type name
    fn name(&self) -> &str;
}
