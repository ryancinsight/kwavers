//! Configuration and builder for `RealTimeSirtPipeline`.
//!
//! SRP: changes when clinical performance constraints, projection model selection,
//! or postprocessing options change.

use super::super::acoustic_projection::AcousticProjectionGeometry;
use crate::solver::inverse::reconstruction::unified_sirt::SirtConfig;

/// Real-time SIRT reconstruction pipeline configuration.
#[derive(Debug, Clone)]
pub struct RealTimeSirtConfig {
    /// Base SIRT algorithm parameters.
    pub sirt_config: SirtConfig,
    /// Target frame rate (fps).
    pub target_frame_rate: f64,
    /// Maximum computation budget per frame (ms).
    pub max_frame_time_ms: f64,
    /// Enable RF data normalization before reconstruction.
    pub enable_preprocessing: bool,
    /// Gaussian smoothing sigma applied to each output frame (grid points).
    pub output_smoothing_sigma: Option<f64>,
    /// Threshold below which output intensities are zeroed.
    pub intensity_threshold: Option<f64>,
    /// Enable per-frame SNR and artifact quality metrics.
    pub enable_quality_monitoring: bool,
    /// Accumulate frames without waiting for a complete dataset.
    pub streaming_mode: bool,
    /// Enable input validation and output verification.
    pub enable_safety_checks: bool,
    /// Physics-based acoustic transducer geometry for SIRT projection.
    ///
    /// `Some(g)` selects the distance-weighted, attenuation-corrected
    /// acoustic model (Dines & Kak 1979).  `None` falls back to the
    /// legacy uniform column-sum model.
    pub transducer_geometry: Option<AcousticProjectionGeometry>,
}

impl Default for RealTimeSirtConfig {
    fn default() -> Self {
        Self {
            sirt_config: SirtConfig::default()
                .with_iterations(10)
                .with_relaxation(0.5),
            target_frame_rate: 10.0,
            max_frame_time_ms: 100.0,
            enable_preprocessing: true,
            output_smoothing_sigma: Some(0.5),
            intensity_threshold: None,
            enable_quality_monitoring: true,
            streaming_mode: true,
            enable_safety_checks: true,
            transducer_geometry: None,
        }
    }
}

impl RealTimeSirtConfig {
    /// High-quality diagnostic mode (slower, more iterations).
    #[must_use]
    pub fn diagnostic_quality(mut self) -> Self {
        self.sirt_config = self.sirt_config.with_iterations(20).with_relaxation(0.4);
        self.max_frame_time_ms = 200.0;
        self.target_frame_rate = 5.0;
        self
    }

    /// High-throughput streaming mode (fewer iterations, higher fps).
    #[must_use]
    pub fn fast_streaming(mut self) -> Self {
        self.sirt_config = self.sirt_config.with_iterations(5).with_relaxation(0.6);
        self.max_frame_time_ms = 50.0;
        self.target_frame_rate = 20.0;
        self
    }

    /// Enable Gaussian output smoothing with `sigma` grid points.
    #[must_use]
    pub fn with_output_smoothing(mut self, sigma: f64) -> Self {
        self.output_smoothing_sigma = Some(sigma);
        self
    }

    /// Zero output intensities below `threshold`.
    #[must_use]
    pub fn with_intensity_threshold(mut self, threshold: f64) -> Self {
        self.intensity_threshold = Some(threshold);
        self
    }

    /// Use the physics-based acoustic projection model with `geometry`.
    #[must_use]
    pub fn with_acoustic_projection(mut self, geometry: AcousticProjectionGeometry) -> Self {
        self.transducer_geometry = Some(geometry);
        self
    }
}
