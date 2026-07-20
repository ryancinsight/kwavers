//! MlConformalPredictor — calibration and interval generation

use super::config::ConformalConfig;
use super::types::{
    CalibrationSummary, ConformalResult, ConformalValidationMetrics, PredictionIntervalBatch,
    ScoreDistribution,
};
#[cfg(feature = "pinn")]
use crate::ml::uncertainty::PinnUncertaintyPredictor;
use kwavers_core::error::KwaversResult;
use leto::Array2;
use log::info;
use std::borrow::Cow;
use std::collections::HashMap;
use tyche_core::{ConformalCalibrator, ConformalError, Moments};

const INTERVAL_CONFIDENCE_LEVELS: [f64; 3] = [0.8, 0.9, 0.95];

/// Conformal predictor for uncertainty quantification
#[derive(Debug)]
pub struct MlConformalPredictor {
    pub(super) config: ConformalConfig,
    calibrator: ConformalCalibrator<f64>,
    calibration_scores: Vec<f64>,
    is_calibrated: bool,
}

impl MlConformalPredictor {
    /// Create new conformal predictor
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new(config: ConformalConfig) -> KwaversResult<Self> {
        if config.calibration_size == 0 {
            return Err(kwavers_core::error::KwaversError::InvalidInput(
                "Calibration size must be non-zero".to_owned(),
            ));
        }
        let calibrator = ConformalCalibrator::new(1.0 - config.confidence_level)
            .map_err(|error| conformal_error("Confidence level is invalid", error))?;

        Ok(Self {
            config,
            calibrator,
            calibration_scores: Vec::new(),
            is_calibrated: false,
        })
    }

    /// Calibrate the conformal predictor using a calibration dataset
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn calibrate(
        &mut self,
        predictions: &[Array2<f32>],
        targets: &[Array2<f32>],
    ) -> KwaversResult<()> {
        if predictions.len() != targets.len() {
            return Err(kwavers_core::error::KwaversError::InvalidInput(
                "Predictions and targets must have same length".to_owned(),
            ));
        }
        if predictions.len() != self.config.calibration_size {
            return Err(kwavers_core::error::KwaversError::InvalidInput(format!(
                "Calibration set has {} samples; configuration requires {}",
                predictions.len(),
                self.config.calibration_size
            )));
        }

        let mut calibration_scores = Vec::with_capacity(predictions.len());
        for (pred, target) in predictions.iter().zip(targets.iter()) {
            calibration_scores.push(self.compute_conformity_score(pred, target)?);
        }
        self.calibrator
            .calibrate_in_place(&mut calibration_scores)
            .map_err(|error| conformal_error("invalid calibration scores", error))?;

        self.calibration_scores = calibration_scores;
        self.is_calibrated = true;
        info!(
            "Conformal predictor calibrated with {} samples",
            self.calibration_scores.len()
        );

        Ok(())
    }

    /// Quantify uncertainty using conformal prediction
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    #[cfg(feature = "pinn")]
    pub fn quantify_uncertainty<P: PinnUncertaintyPredictor + ?Sized>(
        &self,
        predictor: &P,
        inputs: &Array2<f32>,
        ground_truth: Option<&Array2<f32>>,
    ) -> KwaversResult<crate::ml::uncertainty::MlPredictionWithUncertainty> {
        if !self.is_calibrated {
            return Err(kwavers_core::error::KwaversError::InvalidInput(
                "Conformal predictor must be calibrated before use".to_string(),
            ));
        }

        let prediction = predictor.predict_inputs(inputs)?;

        let quantile =
            score_to_prediction_precision(self.compute_quantile(self.config.confidence_level)?);

        let uncertainty = Array2::from_elem(prediction.shape(), quantile);

        let lower_bound = &prediction - &uncertainty;
        let upper_bound = &prediction + &uncertainty;

        let reliability_score = if let Some(target) = ground_truth {
            validate_array_pair(&prediction, target)?;
            let covered = prediction
                .iter()
                .zip(target.iter())
                .filter(|&(&predicted, &observed)| (predicted - observed).abs() <= quantile)
                .count();
            covered as f64 / prediction.len() as f64
        } else {
            self.config.confidence_level
        };

        let mut confidence_intervals = HashMap::new();
        let conf_level_str = format!("{:.0}%", self.config.confidence_level * 100.0);
        confidence_intervals.insert(conf_level_str, (lower_bound, upper_bound));

        Ok(crate::ml::uncertainty::MlPredictionWithUncertainty {
            mean_prediction: prediction,
            uncertainty,
            confidence_intervals,
            reliability_score,
        })
    }

    /// Generate prediction intervals for new data
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn predict_intervals(
        &self,
        predictions: &[Array2<f32>],
    ) -> KwaversResult<ConformalResult<'_>> {
        if !self.is_calibrated {
            return Err(kwavers_core::error::KwaversError::InvalidInput(
                "Predictor must be calibrated".to_owned(),
            ));
        }
        if predictions.is_empty() {
            return Err(kwavers_core::error::KwaversError::InvalidInput(
                "Prediction set must be non-empty".to_owned(),
            ));
        }
        for (sample, prediction) in predictions.iter().enumerate() {
            validate_array_values(prediction, "Prediction", Some(sample))?;
        }

        let mut prediction_intervals = HashMap::new();

        for level in INTERVAL_CONFIDENCE_LEVELS {
            let quantile = score_to_prediction_precision(self.compute_quantile(level)?);
            let lower_bounds = predictions
                .iter()
                .map(|prediction| prediction - quantile)
                .collect();
            let upper_bounds = predictions
                .iter()
                .map(|prediction| prediction + quantile)
                .collect();
            prediction_intervals.insert(
                format!("{:.0}%", level * 100.0),
                PredictionIntervalBatch {
                    lower: lower_bounds,
                    upper: upper_bounds,
                },
            );
        }

        Ok(ConformalResult {
            prediction_intervals,
            target_coverage_probability: self.config.confidence_level,
            conformity_scores: Cow::Borrowed(&self.calibration_scores),
        })
    }

    /// Validate conformal prediction performance
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn validate_performance(
        &self,
        test_predictions: &[Array2<f32>],
        test_targets: &[Array2<f32>],
    ) -> KwaversResult<ConformalValidationMetrics> {
        if !self.is_calibrated {
            return Err(kwavers_core::error::KwaversError::InvalidInput(
                "Predictor must be calibrated".to_owned(),
            ));
        }
        if test_predictions.len() != test_targets.len() {
            return Err(kwavers_core::error::KwaversError::InvalidInput(
                "Test predictions and targets must have same length".to_owned(),
            ));
        }
        if test_predictions.is_empty() {
            return Err(kwavers_core::error::KwaversError::InvalidInput(
                "Validation set must be non-empty".to_owned(),
            ));
        }

        let mut coverage_count = 0;
        let quantile =
            score_to_prediction_precision(self.compute_quantile(self.config.confidence_level)?);

        for (pred, target) in test_predictions.iter().zip(test_targets.iter()) {
            validate_array_pair(pred, target)?;
            let target_within_interval = pred
                .iter()
                .zip(target.iter())
                .all(|(&prediction, &observed)| (prediction - observed).abs() <= quantile);

            if target_within_interval {
                coverage_count += 1;
            }
        }

        let empirical_coverage = coverage_count as f64 / test_predictions.len() as f64;
        let mean_interval_width = 2.0 * quantile;
        let coverage_efficiency = (mean_interval_width > 0.0)
            .then_some(empirical_coverage / f64::from(mean_interval_width));

        Ok(ConformalValidationMetrics {
            empirical_coverage,
            target_coverage: self.config.confidence_level,
            mean_interval_width,
            coverage_efficiency,
        })
    }

    /// Check if predictor is calibrated
    #[must_use]
    pub fn is_calibrated(&self) -> bool {
        self.is_calibrated
    }

    /// Get calibration summary
    #[must_use]
    pub fn calibration_summary(&self) -> CalibrationSummary {
        if !self.is_calibrated {
            return CalibrationSummary {
                is_calibrated: false,
                num_calibration_samples: 0,
                score_distribution: None,
            };
        }

        let min_score = self
            .calibration_scores
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b));
        let max_score = self
            .calibration_scores
            .iter()
            .fold(0.0_f64, |a, &b| a.max(b));
        let mut moments = Moments::new();
        for &score in &self.calibration_scores {
            moments.update(score);
        }
        let mean_score = moments
            .mean()
            .expect("invariant: successful calibration stores at least one score");

        let median_score = if self.calibration_scores.len().is_multiple_of(2) {
            let mid = self.calibration_scores.len() / 2;
            (self.calibration_scores[mid - 1] + self.calibration_scores[mid]) / 2.0
        } else {
            self.calibration_scores[self.calibration_scores.len() / 2]
        };

        CalibrationSummary {
            is_calibrated: true,
            num_calibration_samples: self.calibration_scores.len(),
            score_distribution: Some(ScoreDistribution {
                min_score,
                max_score,
                mean_score,
                median_score,
            }),
        }
    }

    fn compute_conformity_score(
        &self,
        prediction: &Array2<f32>,
        target: &Array2<f32>,
    ) -> KwaversResult<f64> {
        validate_array_pair(prediction, target)?;
        let mut errors = Vec::with_capacity(prediction.len());
        for (&predicted, &observed) in prediction.iter().zip(target.iter()) {
            errors.push((predicted - observed).abs());
        }
        errors.sort_by(|a, b| a.total_cmp(b));

        let mid = errors.len() / 2;
        let median = if errors.len().is_multiple_of(2) {
            (errors[mid - 1] + errors[mid]) / 2.0
        } else {
            errors[mid]
        };
        Ok(f64::from(median))
    }

    fn compute_quantile(&self, confidence_level: f64) -> KwaversResult<f64> {
        ConformalCalibrator::new(1.0 - confidence_level)
            .map_err(|error| conformal_error("Confidence level is invalid", error))?
            .calibrate_sorted(&self.calibration_scores)
            .map_err(|error| conformal_error("invalid calibration scores", error))
    }
}

fn validate_array_pair(prediction: &Array2<f32>, target: &Array2<f32>) -> KwaversResult<()> {
    if prediction.shape() != target.shape() {
        return Err(kwavers_core::error::KwaversError::InvalidInput(format!(
            "Prediction shape {:?} does not match target shape {:?}",
            prediction.shape(),
            target.shape()
        )));
    }
    if prediction.is_empty() {
        return Err(kwavers_core::error::KwaversError::InvalidInput(
            "Conformity arrays must be non-empty".to_owned(),
        ));
    }
    validate_array_values(prediction, "Prediction", None)?;
    validate_array_values(target, "Target", None)?;
    Ok(())
}

fn validate_array_values(
    array: &Array2<f32>,
    context: &str,
    sample: Option<usize>,
) -> KwaversResult<()> {
    if array.is_empty() {
        let sample = sample.map_or_else(String::new, |value| format!(" sample {value}"));
        return Err(kwavers_core::error::KwaversError::InvalidInput(format!(
            "{context}{sample} must be non-empty"
        )));
    }
    if let Some((index, value)) = array
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        let sample = sample.map_or_else(String::new, |value| format!(" sample {value}"));
        return Err(kwavers_core::error::KwaversError::InvalidInput(format!(
            "{context}{sample} contains non-finite value at element {index}: {value}"
        )));
    }
    Ok(())
}

fn conformal_error<T: std::fmt::Debug>(
    context: &str,
    error: ConformalError<T>,
) -> kwavers_core::error::KwaversError {
    kwavers_core::error::KwaversError::InvalidInput(format!("{context}: {error}"))
}

fn score_to_prediction_precision(score: f64) -> f32 {
    let narrowed = score as f32;
    debug_assert_eq!(
        f64::from(narrowed).to_bits(),
        score.to_bits(),
        "invariant: every Analysis score is widened from f32 and Tyche selects an existing score"
    );
    narrowed
}

#[cfg(test)]
mod tests;
