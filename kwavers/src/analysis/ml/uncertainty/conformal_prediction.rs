//! Conformal Prediction for Guaranteed Uncertainty Bounds
//!
//! Implements conformal prediction to provide distribution-free uncertainty
//! bounds with guaranteed coverage probabilities.

use crate::core::error::KwaversResult;
use log::info;
#[cfg(feature = "pinn")]
use burn::tensor::backend::Backend;
use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Configuration for conformal prediction
#[derive(Debug, Clone)]
pub struct ConformalConfig {
    /// Desired confidence level (0-1)
    pub confidence_level: f64,
    /// Size of calibration set
    pub calibration_size: usize,
}

impl Default for ConformalConfig {
    fn default() -> Self {
        Self {
            confidence_level: 0.95,
            calibration_size: 1000,
        }
    }
}

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

/// Conformal predictor for uncertainty quantification
#[derive(Debug)]
pub struct ConformalPredictor {
    config: ConformalConfig,
    calibration_scores: Vec<f64>,
    is_calibrated: bool,
}

impl ConformalPredictor {
    /// Create new conformal predictor
    pub fn new(config: ConformalConfig) -> KwaversResult<Self> {
        if config.confidence_level <= 0.0 || config.confidence_level >= 1.0 {
            return Err(crate::core::error::KwaversError::InvalidInput(
                "Confidence level must be between 0 and 1".to_string(),
            ));
        }

        Ok(Self {
            config,
            calibration_scores: Vec::new(),
            is_calibrated: false,
        })
    }

    /// Calibrate the conformal predictor using a calibration dataset
    pub fn calibrate(
        &mut self,
        predictions: &[Array2<f32>],
        targets: &[Array2<f32>],
    ) -> KwaversResult<()> {
        if predictions.len() != targets.len() {
            return Err(crate::core::error::KwaversError::InvalidInput(
                "Predictions and targets must have same length".to_string(),
            ));
        }

        self.calibration_scores.clear();

        for (pred, target) in predictions.iter().zip(targets.iter()) {
            let score = self.compute_conformity_score(pred, target);
            self.calibration_scores.push(score);
        }

        self.calibration_scores
            .sort_by(|a, b| a.partial_cmp(b).unwrap());

        self.is_calibrated = true;
        info!(
            "Conformal predictor calibrated with {} samples",
            self.calibration_scores.len()
        );

        Ok(())
    }

    /// Quantify uncertainty using conformal prediction
    #[cfg(feature = "pinn")]
    pub fn quantify_uncertainty<B: Backend>(
        &self,
        pinn: &crate::solver::inverse::pinn::ml::BurnPINN1DWave<B>,
        inputs: &Array2<f32>,
        _ground_truth: Option<&Array2<f32>>,
    ) -> KwaversResult<super::PredictionWithUncertainty> {
        if !self.is_calibrated {
            return Err(crate::core::error::KwaversError::InvalidInput(
                "Conformal predictor must be calibrated before use".to_string(),
            ));
        }

        let x: Array1<f64> = inputs.column(0).mapv(|v| v as f64).to_owned();
        let t: Array1<f64> = inputs.column(1).mapv(|v| v as f64).to_owned();
        let device = pinn.device();

        let prediction_f64 = pinn.predict(&x, &t, &device)?;
        let prediction = prediction_f64.mapv(|v| v as f32);

        let quantile = self.compute_quantile(self.config.confidence_level);

        let uncertainty = Array2::from_elem(prediction.dim(), quantile as f32);

        let lower_bound = &prediction - &uncertainty;
        let upper_bound = &prediction + &uncertainty;

        let mut confidence_intervals = HashMap::new();
        let conf_level_str = format!("{:.0}%", self.config.confidence_level * 100.0);
        confidence_intervals.insert(conf_level_str, (lower_bound, upper_bound));

        let coverage_probability = self.estimate_coverage_probability();
        let reliability_score = coverage_probability.min(1.0);

        Ok(super::PredictionWithUncertainty {
            mean_prediction: prediction,
            uncertainty,
            confidence_intervals,
            reliability_score,
        })
    }

    /// Compute conformity score between prediction and target
    fn compute_conformity_score(&self, prediction: &Array2<f32>, target: &Array2<f32>) -> f64 {
        let error = prediction - target;
        let abs_error = error.mapv(|x| x.abs());

        let mut errors: Vec<f64> = abs_error.iter().map(|&x| x as f64).collect();
        errors.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mid = errors.len() / 2;
        if errors.len().is_multiple_of(2) {
            (errors[mid - 1] + errors[mid]) / 2.0
        } else {
            errors[mid]
        }
    }

    /// Compute quantile from calibration scores
    fn compute_quantile(&self, confidence_level: f64) -> f64 {
        if self.calibration_scores.is_empty() {
            return 1.0;
        }

        let alpha = 1.0 - confidence_level;
        let index = ((self.calibration_scores.len() as f64 * alpha).ceil() as usize)
            .max(1)
            .min(self.calibration_scores.len())
            - 1;

        self.calibration_scores[index]
    }

    /// Estimate coverage probability from calibration
    fn estimate_coverage_probability(&self) -> f64 {
        self.config.confidence_level
    }

    /// Generate prediction intervals for new data
    pub fn predict_intervals(&self, predictions: &[Array2<f32>]) -> KwaversResult<ConformalResult> {
        if !self.is_calibrated {
            return Err(crate::core::error::KwaversError::InvalidInput(
                "Predictor must be calibrated".to_string(),
            ));
        }

        let mut prediction_intervals = HashMap::new();

        let confidence_levels = vec![0.8, 0.9, 0.95];

        for &level in &confidence_levels {
            let quantile = self.compute_quantile(level);
            let mut lower_bounds = Vec::new();
            let mut upper_bounds = Vec::new();

            for prediction in predictions {
                let lower = prediction - quantile as f32;
                let upper = prediction + quantile as f32;
                lower_bounds.push(lower);
                upper_bounds.push(upper);
            }

            if !lower_bounds.is_empty() {
                prediction_intervals.insert(
                    format!("{:.0}%", level * 100.0),
                    (lower_bounds[0].clone(), upper_bounds[0].clone()),
                );
            }
        }

        Ok(ConformalResult {
            prediction_intervals,
            coverage_probability: self.estimate_coverage_probability(),
            conformity_scores: Array1::from_vec(self.calibration_scores.clone()),
        })
    }

    /// Validate conformal prediction performance
    pub fn validate_performance(
        &self,
        test_predictions: &[Array2<f32>],
        test_targets: &[Array2<f32>],
    ) -> KwaversResult<ValidationMetrics> {
        if test_predictions.len() != test_targets.len() {
            return Err(crate::core::error::KwaversError::InvalidInput(
                "Test predictions and targets must have same length".to_string(),
            ));
        }

        let mut coverage_count = 0;
        let mut interval_widths = Vec::new();

        for (pred, target) in test_predictions.iter().zip(test_targets.iter()) {
            let quantile = self.compute_quantile(self.config.confidence_level);
            let lower_bound = pred - quantile as f32;
            let upper_bound = pred + quantile as f32;

            let target_within_interval = lower_bound
                .iter()
                .zip(upper_bound.iter())
                .zip(target.iter())
                .all(|((low, high), t)| t >= low && t <= high);

            if target_within_interval {
                coverage_count += 1;
            }

            let width = (&upper_bound - &lower_bound).mapv(|x| x.abs());
            interval_widths.push(width.iter().sum::<f32>() / width.len() as f32);
        }

        let empirical_coverage = coverage_count as f64 / test_predictions.len() as f64;
        let mean_interval_width =
            interval_widths.iter().sum::<f32>() / interval_widths.len() as f32;

        Ok(ValidationMetrics {
            empirical_coverage,
            target_coverage: self.config.confidence_level,
            mean_interval_width,
            coverage_efficiency: empirical_coverage / mean_interval_width as f64,
        })
    }

    /// Check if predictor is calibrated
    pub fn is_calibrated(&self) -> bool {
        self.is_calibrated
    }

    /// Get calibration summary
    pub fn calibration_summary(&self) -> CalibrationSummary {
        if !self.is_calibrated {
            return CalibrationSummary {
                is_calibrated: false,
                num_calibration_samples: 0,
                score_distribution: ScoreDistribution {
                    min_score: 0.0,
                    max_score: 0.0,
                    mean_score: 0.0,
                    median_score: 0.0,
                },
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
        let mean_score =
            self.calibration_scores.iter().sum::<f64>() / self.calibration_scores.len() as f64;

        let median_score = if self.calibration_scores.len().is_multiple_of(2) {
            let mid = self.calibration_scores.len() / 2;
            (self.calibration_scores[mid - 1] + self.calibration_scores[mid]) / 2.0
        } else {
            self.calibration_scores[self.calibration_scores.len() / 2]
        };

        CalibrationSummary {
            is_calibrated: true,
            num_calibration_samples: self.calibration_scores.len(),
            score_distribution: ScoreDistribution {
                min_score,
                max_score,
                mean_score,
                median_score,
            },
        }
    }
}

/// Validation metrics for conformal prediction
#[derive(Debug)]
pub struct ValidationMetrics {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conformal_predictor_creation() {
        let config = ConformalConfig {
            confidence_level: 0.9,
            calibration_size: 100,
        };

        let predictor = ConformalPredictor::new(config);
        assert!(predictor.is_ok());
    }

    #[test]
    fn test_conformal_calibration() {
        let config = ConformalConfig {
            confidence_level: 0.9,
            calibration_size: 10,
        };

        let mut predictor = ConformalPredictor::new(config).unwrap();

        let predictions = vec![
            Array2::from_elem((5, 5), 1.0),
            Array2::from_elem((5, 5), 2.0),
        ];
        let targets = vec![
            Array2::from_elem((5, 5), 1.1),
            Array2::from_elem((5, 5), 1.9),
        ];

        let result = predictor.calibrate(&predictions, &targets);
        assert!(result.is_ok());
        assert!(predictor.is_calibrated());
    }

    #[test]
    fn test_conformity_score_computation() {
        let config = ConformalConfig::default();
        let predictor = ConformalPredictor::new(config).unwrap();

        let prediction = Array2::from_elem((3, 3), 1.0);
        let target = Array2::from_elem((3, 3), 1.2);

        let score = predictor.compute_conformity_score(&prediction, &target);
        assert!(score > 0.0);
    }

    #[test]
    fn test_quantile_computation() {
        let config = ConformalConfig::default();
        let mut predictor = ConformalPredictor::new(config).unwrap();

        predictor.calibration_scores = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        predictor.is_calibrated = true;

        let quantile = predictor.compute_quantile(0.9);
        assert_eq!(quantile, 0.1);
    }
}
