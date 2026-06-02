use serde::{Deserialize, Serialize};

/// Methods for separating clutter and blood subspaces
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SubspaceSeparationMethod {
    /// Fixed rank cutoff (e.g., first N eigenmodes are clutter)
    FixedRank { clutter_rank: usize },
    /// Adaptive threshold based on eigenvalue decay
    AdaptiveThreshold { decay_factor: f64 },
    /// CBR-based automatic selection
    CbrBased { target_cbr_db: f64 },
}

/// Methods for estimating clutter-to-blood ratio
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CbrEstimationMethod {
    /// Sum of eigenvalues (simple, fast)
    EigenvalueSum,
    /// Power ratio between subspaces
    PowerRatio,
    /// Maximum likelihood estimation
    MaximumLikelihood,
}

/// Configuration for adaptive clutter filter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveFilterConfig {
    /// Method for clutter/blood subspace separation
    pub separation_method: SubspaceSeparationMethod,
    /// CBR estimation method
    pub cbr_estimation: CbrEstimationMethod,
    /// Minimum eigenvalue threshold (relative to maximum eigenvalue)
    pub noise_floor_threshold: f64,
    /// Enable temporal smoothing of CBR estimates across ensembles
    pub temporal_smoothing: bool,
    /// Smoothing window size (only used if temporal_smoothing = true)
    pub smoothing_window: usize,
}

impl Default for AdaptiveFilterConfig {
    fn default() -> Self {
        Self {
            separation_method: SubspaceSeparationMethod::AdaptiveThreshold { decay_factor: 0.1 },
            cbr_estimation: CbrEstimationMethod::EigenvalueSum,
            noise_floor_threshold: 1e-6,
            temporal_smoothing: false,
            smoothing_window: 3,
        }
    }
}
