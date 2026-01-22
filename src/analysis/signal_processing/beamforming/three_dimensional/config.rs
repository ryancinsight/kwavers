//! Configuration types for 3D beamforming operations
//!
//! This module defines the configuration structures and enums for 3D beamforming,
//! including algorithm selection, apodization windows, and performance metrics.

use crate::domain::sensor::beamforming::BeamformingConfig;

/// 3D beamforming algorithm types optimized for volumetric imaging
#[derive(Debug, Clone)]
pub enum BeamformingAlgorithm3D {
    /// Delay-and-Sum with dynamic focusing and apodization
    DelayAndSum {
        /// Dynamic focusing enabled
        dynamic_focusing: bool,
        /// Apodization window type
        apodization: ApodizationWindow,
        /// Sub-volume processing for memory efficiency
        sub_volume_size: Option<(usize, usize, usize)>,
    },
    /// Minimum Variance Distortionless Response (MVDR) for 3D
    MVDR3D {
        /// Diagonal loading factor
        diagonal_loading: f64,
        /// Subarray size for covariance estimation
        subarray_size: [usize; 3],
    },
    /// Synthetic Aperture Focusing Technique (SAFT) for 3D
    SAFT3D {
        /// Virtual source density
        virtual_sources: usize,
    },
}

/// Apodization windows for sidelobe reduction in 3D beamforming
#[derive(Debug, Clone)]
pub enum ApodizationWindow {
    /// Rectangular window (no apodization)
    Rectangular,
    /// Hamming window
    Hamming,
    /// Hann window
    Hann,
    /// Blackman window
    Blackman,
    /// Gaussian window with specified sigma
    Gaussian { sigma: f64 },
    /// Custom window function
    Custom(Vec<f64>),
}

impl ApodizationWindow {
    /// Convert window type to u32 identifier for GPU shaders
    #[must_use]
    pub fn to_shader_id(&self) -> u32 {
        match self {
            ApodizationWindow::Rectangular => 0,
            ApodizationWindow::Hamming => 1,
            ApodizationWindow::Hann => 2,
            ApodizationWindow::Blackman => 3,
            ApodizationWindow::Gaussian { .. } => 4,
            ApodizationWindow::Custom(_) => 5,
        }
    }
}

/// Configuration for 3D beamforming operations
#[derive(Debug, Clone)]
pub struct BeamformingConfig3D {
    /// Base 2D configuration
    pub base_config: BeamformingConfig,
    /// Volume dimensions (nx, ny, nz)
    pub volume_dims: (usize, usize, usize),
    /// Voxel spacing in meters (dx, dy, dz)
    pub voxel_spacing: (f64, f64, f64),
    /// Number of transducer elements in 3D array
    pub num_elements_3d: (usize, usize, usize),
    /// Element spacing in 3D (sx, sy, sz)
    pub element_spacing_3d: (f64, f64, f64),
    /// Center frequency for beamforming (Hz)
    pub center_frequency: f64,
    /// Sampling frequency (Hz)
    pub sampling_frequency: f64,
    /// Sound speed in tissue (m/s)
    pub sound_speed: f64,
    /// GPU device selection
    pub gpu_device: Option<String>,
    /// Enable real-time streaming
    pub enable_streaming: bool,
    /// Streaming buffer size (number of frames)
    pub streaming_buffer_size: usize,
}

impl Default for BeamformingConfig3D {
    fn default() -> Self {
        Self {
            base_config: BeamformingConfig::default(),
            volume_dims: (128, 128, 128),
            voxel_spacing: (0.5e-3, 0.5e-3, 0.5e-3), // 0.5mm isotropic voxels
            num_elements_3d: (32, 32, 16),           // 32x32x16 = 16,384 elements
            element_spacing_3d: (0.3e-3, 0.3e-3, 0.5e-3), // λ/2 spacing at ~2.5MHz
            center_frequency: 2.5e6,
            sampling_frequency: 50e6,
            sound_speed: 1540.0,
            gpu_device: None,
            enable_streaming: true,
            streaming_buffer_size: 16,
        }
    }
}

/// Performance metrics for beamforming operations
#[derive(Debug, Default)]
pub struct BeamformingMetrics {
    /// Processing time per volume (ms)
    pub processing_time_ms: f64,
    /// GPU memory usage (MB)
    pub gpu_memory_mb: f64,
    /// CPU memory usage (MB)
    pub cpu_memory_mb: f64,
    /// Reconstruction rate (volumes/second)
    pub reconstruction_rate: f64,
    /// Dynamic range achieved (dB)
    pub dynamic_range_db: f64,
    /// Signal-to-noise ratio (dB)
    pub snr_db: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beamforming_config_3d_default() {
        let config = BeamformingConfig3D::default();
        assert_eq!(config.volume_dims, (128, 128, 128));
        assert_eq!(config.num_elements_3d, (32, 32, 16));
        assert_eq!(config.center_frequency, 2.5e6);
    }

    #[test]
    fn test_metrics_default() {
        let metrics = BeamformingMetrics::default();
        assert_eq!(metrics.processing_time_ms, 0.0);
        assert_eq!(metrics.gpu_memory_mb, 0.0);
        assert_eq!(metrics.reconstruction_rate, 0.0);
    }

    #[test]
    fn test_algorithm_variants() {
        let _das = BeamformingAlgorithm3D::DelayAndSum {
            dynamic_focusing: true,
            apodization: ApodizationWindow::Hamming,
            sub_volume_size: Some((64, 64, 64)),
        };

        let _mvdr = BeamformingAlgorithm3D::MVDR3D {
            diagonal_loading: 0.01,
            subarray_size: [8, 8, 4],
        };

        let _saft = BeamformingAlgorithm3D::SAFT3D {
            virtual_sources: 100,
        };
    }

    #[test]
    fn test_apodization_window_variants() {
        let _rect = ApodizationWindow::Rectangular;
        let _hamming = ApodizationWindow::Hamming;
        let _hann = ApodizationWindow::Hann;
        let _blackman = ApodizationWindow::Blackman;
        let _gaussian = ApodizationWindow::Gaussian { sigma: 0.5 };
        let _custom = ApodizationWindow::Custom(vec![1.0, 0.8, 0.6]);
    }

    #[test]
    fn test_apodization_window_shader_id() {
        assert_eq!(ApodizationWindow::Rectangular.to_shader_id(), 0);
        assert_eq!(ApodizationWindow::Hamming.to_shader_id(), 1);
        assert_eq!(ApodizationWindow::Hann.to_shader_id(), 2);
        assert_eq!(ApodizationWindow::Blackman.to_shader_id(), 3);
        assert_eq!(ApodizationWindow::Gaussian { sigma: 0.5 }.to_shader_id(), 4);
        assert_eq!(ApodizationWindow::Custom(vec![]).to_shader_id(), 5);
    }
}
