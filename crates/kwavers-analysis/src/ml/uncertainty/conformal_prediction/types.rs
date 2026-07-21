//! Result and summary types for conformal prediction.

use leto::Array2;
use std::borrow::Cow;
use std::collections::HashMap;

/// Conformal prediction result for every supplied prediction.
#[derive(Debug)]
#[non_exhaustive]
pub struct ConformalResult<'scores> {
    /// Lower and upper arrays for every prediction, keyed by confidence level.
    pub prediction_intervals: HashMap<String, PredictionIntervalBatch>,
    /// Requested marginal coverage probability, represented by one batch's
    /// exact [`PredictionIntervalBatch::confidence_level`].
    pub target_coverage_probability: f64,
    /// Borrowed conformity scores from calibration.
    pub conformity_scores: Cow<'scores, [f64]>,
}

/// Aligned lower and upper arrays for one confidence level.
#[derive(Debug)]
#[non_exhaustive]
pub struct PredictionIntervalBatch {
    /// Exact marginal coverage probability represented by this batch.
    pub confidence_level: f64,
    /// Lower endpoint arrays in input order.
    pub lower: Vec<Array2<f32>>,
    /// Upper endpoint arrays in input order.
    pub upper: Vec<Array2<f32>>,
}

/// Validation metrics for conformal prediction.
#[derive(Debug)]
#[non_exhaustive]
pub struct ConformalValidationMetrics {
    /// Empirical coverage probability achieved.
    pub empirical_coverage: f64,
    /// Target coverage probability.
    pub target_coverage: f64,
    /// Mean prediction interval width.
    pub mean_interval_width: f32,
    /// Coverage divided by width, or `None` for a zero-width interval.
    pub coverage_efficiency: Option<f64>,
}

/// Calibration summary.
#[derive(Debug)]
#[non_exhaustive]
pub struct CalibrationSummary {
    /// Whether calibration has completed.
    pub is_calibrated: bool,
    /// Number of calibration samples.
    pub num_calibration_samples: usize,
    /// Score distribution, absent before calibration.
    pub score_distribution: Option<ScoreDistribution>,
}

/// Distribution of conformity scores.
#[derive(Debug)]
#[non_exhaustive]
pub struct ScoreDistribution {
    /// Minimum score.
    pub min_score: f64,
    /// Maximum score.
    pub max_score: f64,
    /// Arithmetic mean.
    pub mean_score: f64,
    /// Median score.
    pub median_score: f64,
}
