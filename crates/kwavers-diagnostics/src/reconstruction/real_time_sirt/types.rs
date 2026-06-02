//! Frame-level result and quality types for real-time SIRT reconstruction.
//!
//! SRP: changes when the output schema for a single reconstruction frame changes.

use ndarray::Array3;

/// Reconstructed image for a single RF measurement frame.
#[derive(Debug, Clone)]
pub struct ReconstructionFrame {
    /// Seconds since pipeline start.
    pub timestamp: f64,
    /// Reconstructed image volume.
    pub image: Array3<f64>,
    /// Number of SIRT iterations performed.
    pub iterations: usize,
    /// Wall-clock computation time (ms).
    pub computation_time_ms: f64,
    /// Relative residual norm ‖r‖₂/‖b‖₂ after the final iteration.
    pub convergence_error: f64,
    /// Per-frame quality metrics (populated when quality monitoring is enabled).
    pub quality_metrics: Option<FrameQuality>,
}

/// Per-frame quality assessment metrics.
#[derive(Debug, Clone)]
pub struct FrameQuality {
    /// Estimated SNR (dB).
    pub snr_estimate: f64,
    /// Artifact presence indicator: 0.0 = none, 1.0 = severe.
    pub artifact_level: f64,
    /// Spatial smoothness measure (lower = more edge structure).
    pub spatial_smoothness: f64,
    /// Intensity dynamic range (max − min).
    pub dynamic_range: f64,
    /// Whether the SIRT residual met the convergence criterion.
    pub converged: bool,
}
