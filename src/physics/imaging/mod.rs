//! Imaging physics module
//!
//! This module provides various imaging modalities for acoustic and optical imaging.

pub mod elastography;
pub mod photoacoustic;
pub mod seismic;
pub mod ultrasound;

/// Imaging modality types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ImagingModality {
    /// Photoacoustic imaging
    Photoacoustic,
    /// Thermoacoustic imaging
    Thermoacoustic,
    /// Full waveform inversion
    FullWaveformInversion,
    /// Reverse time migration
    ReverseTimeMigration,
    /// B-mode ultrasound
    BMode,
    /// Doppler imaging
    Doppler,
    /// Elastography
    Elastography,
    /// Acoustic tomography
    AcousticTomography,
}

/// Common imaging configuration
#[derive(Debug, Clone)]
pub struct ImagingConfig {
    /// Selected modality
    pub modality: ImagingModality,
    /// Spatial resolution (m)
    pub resolution: f64,
    /// Field of view (m)
    pub field_of_view: [f64; 3],
    /// Reconstruction algorithm
    pub reconstruction: ReconstructionMethod,
}

/// Reconstruction methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReconstructionMethod {
    /// Time reversal
    TimeReversal,
    /// Backprojection
    Backprojection,
    /// Iterative (e.g., LSQR, TV)
    Iterative,
    /// Model-based
    ModelBased,
    /// Machine learning
    MachineLearning,
}

impl Default for ImagingConfig {
    fn default() -> Self {
        Self {
            modality: ImagingModality::Photoacoustic,
            resolution: 0.1e-3,
            field_of_view: [0.05, 0.05, 0.05],
            reconstruction: ReconstructionMethod::TimeReversal,
        }
    }
}

// Re-export commonly used types
pub use elastography::{ElasticityMap, InversionMethod, ShearWaveElastography};
pub use photoacoustic::{compute_initial_pressure, PhotoacousticConfig};
pub use seismic::{SeismicConfig, SeismicMethod};
pub use ultrasound::{UltrasoundConfig, UltrasoundMode};
