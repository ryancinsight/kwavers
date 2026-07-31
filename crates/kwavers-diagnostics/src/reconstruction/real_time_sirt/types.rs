//! Frame-level result and quality types for real-time SIRT reconstruction.
//!
//! SRP: changes when the output schema for a single reconstruction frame changes.

use aequitas::systems::si::quantities::{Dimensionless, Time};
use leto::Array3;

/// Reconstructed image for a single RF measurement frame.
#[derive(Debug, Clone)]
pub struct ReconstructionFrame {
    /// Elapsed time since pipeline start.
    pub timestamp: Time,
    /// Reconstructed image volume.
    pub image: Array3<f64>,
    /// Number of SIRT iterations performed.
    pub iterations: usize,
    /// Wall-clock computation time.
    pub computation_time: Time,
    /// Relative residual norm ‖r‖₂/‖b‖₂ after the final iteration.
    pub convergence_error: Dimensionless,
    /// Per-frame quality metrics (populated when quality monitoring is enabled).
    pub quality_metrics: Option<FrameQuality>,
}

/// Per-frame quality assessment metrics.
#[derive(Debug, Clone)]
pub struct FrameQuality {
    /// Estimated SNR as a dimensionless logarithmic ratio.
    pub snr_estimate: Dimensionless,
    /// Artifact presence indicator: 0.0 = none, 1.0 = severe.
    pub artifact_level: Dimensionless,
    /// Spatial smoothness measure (lower = more edge structure).
    pub spatial_smoothness: Dimensionless,
    /// Intensity dynamic range (max − min).
    pub dynamic_range: Dimensionless,
    /// Whether the SIRT residual met the convergence criterion.
    pub converged: bool,
}
