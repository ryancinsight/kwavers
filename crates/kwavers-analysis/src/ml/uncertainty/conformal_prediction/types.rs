//! Result and summary types for conformal prediction

use leto::{Array1, Array2};
use std::collections::HashMap;

/// Conformal prediction result
#[derive(Debug)]
pub struct ConformalResult {
    /// Prediction intervals for each point
    pub prediction_intervals: HashMap<String, (Array2<f32>, Array2<f32>)>,
    /// Coverage probability achieved
    pub coverage_probability: f64,
    /// Conformity scores from calibration
    pub conformity_scores: Array1<f64>,
}

/// Validation metrics for conformal prediction
#[derive(Debug)]
pub struct ConformalValidationMetrics {
    /// Empirical coverage probability achieved
    pub empirical_coverage: f64,
    /// Target coverage probability
    pub target_coverage: f64,
    /// Mean prediction interval width
    pub mean_interval_width: f32,
    /// Coverage efficiency (coverage / width)
    pub coverage_efficiency: f64,
}

/// Calibration summary
#[derive(Debug)]
pub struct CalibrationSummary {
    pub is_calibrated: bool,
    pub num_calibration_samples: usize,
    pub score_distribution: ScoreDistribution,
}

/// Distribution of conformity scores
#[derive(Debug)]
pub struct ScoreDistribution {
    pub min_score: f64,
    pub max_score: f64,
    pub mean_score: f64,
    pub median_score: f64,
}
