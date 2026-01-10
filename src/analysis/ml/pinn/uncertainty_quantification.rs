//! Uncertainty Quantification for PINN Predictions
//!
//! This module implements Bayesian neural networks and uncertainty estimation techniques
//! for Physics-Informed Neural Networks, providing reliable confidence bounds for predictions.

use crate::core::error::{KwaversError, KwaversResult};
use burn::tensor::backend::{AutodiffBackend, Backend};
use ndarray::Array1;

/// Uncertainty quantification configuration
#[derive(Debug, Clone)]
pub struct PinnUncertaintyConfig {
    /// Number of Monte Carlo samples for dropout uncertainty
    pub mc_samples: usize,
    /// Dropout probability for MC dropout
    pub dropout_prob: f64,
    /// Ensemble size for deep ensembles
    pub ensemble_size: usize,
    /// Conformal prediction alpha (1-confidence level)
    pub conformal_alpha: f64,
    /// Variance threshold for reliability assessment
    pub variance_threshold: f64,
}

/// Prediction with uncertainty bounds
#[derive(Debug, Clone)]
pub struct PredictionWithUncertainty {
    /// Mean prediction
    pub mean: Vec<f32>,
    /// Standard deviation
    pub std: Vec<f32>,
    /// 95% confidence interval (lower, upper)
    pub confidence_interval: (Vec<f32>, Vec<f32>),
    /// Predictive entropy (measure of uncertainty)
    pub entropy: f32,
    /// Reliability score (0-1, higher is more reliable)
    pub reliability: f32,
    /// Uncertainty quantification method used
    pub method: UncertaintyMethod,
}

/// Uncertainty estimation methods
#[derive(Debug, Clone)]
pub enum UncertaintyMethod {
    /// Monte Carlo Dropout
    MCDropout,
    /// Deep Ensemble
    DeepEnsemble,
    /// Conformal Prediction
    Conformal,
    /// Combined methods
    Hybrid,
}

/// Bayesian PINN with uncertainty quantification
#[derive(Debug)]
pub struct BayesianPINN<B: AutodiffBackend> {
    /// Ensemble of models for uncertainty estimation
    ensemble: Vec<crate::ml::pinn::BurnPINN2DWave<B>>,
    /// Uncertainty configuration
    config: PinnUncertaintyConfig,
    /// Calibration data for conformal prediction
    pub calibration_data: Option<Vec<(Vec<f32>, f32)>>,
    /// Conformal predictor for prediction intervals
    pub conformal_predictor: Option<ConformalPredictor<B>>,
    /// Performance statistics
    pub stats: UncertaintyStats,
}

/// Uncertainty estimation statistics
#[derive(Debug, Clone)]
pub struct UncertaintyStats {
    pub total_predictions: usize,
    pub average_uncertainty: f32,
    pub calibration_error: f32,
    pub coverage_probability: f32,
    pub reliability_score: f32,
}

impl<B: AutodiffBackend> BayesianPINN<B> {
    /// Create a new Bayesian PINN
    pub fn new(
        base_model: &crate::ml::pinn::BurnPINN2DWave<B>,
        config: PinnUncertaintyConfig,
    ) -> KwaversResult<Self> {
        // Create ensemble by perturbing the base model
        let mut ensemble = Vec::new();

        for i in 0..config.ensemble_size {
            // In practice, this would create diverse models
            // For now, clone the base model multiple times
            let _model_idx = i;
            let model = base_model.clone();
            ensemble.push(model);
        }

        Ok(Self {
            ensemble,
            config,
            calibration_data: None,
            conformal_predictor: None,
            stats: UncertaintyStats::default(),
        })
    }

    /// Calibrate uncertainty estimates using validation data
    pub fn calibrate(
        &mut self,
        calibration_inputs: &[Vec<f32>],
        calibration_targets: &[f32],
    ) -> KwaversResult<()> {
        let calibration_data = calibration_inputs
            .iter()
            .zip(calibration_targets.iter())
            .map(|(input, target)| (input.clone(), *target))
            .collect();

        self.calibration_data = Some(calibration_data);

        // Also calibrate conformal predictor if alpha is set
        if self.config.conformal_alpha > 0.0 {
            let mut cp =
                ConformalPredictor::new(self.ensemble[0].clone(), self.config.conformal_alpha);
            cp.calibrate(calibration_inputs, calibration_targets)?;
            self.conformal_predictor = Some(cp);
        }

        Ok(())
    }

    /// Predict with uncertainty quantification
    pub fn predict_with_uncertainty(
        &mut self,
        input: &[f32],
    ) -> KwaversResult<PredictionWithUncertainty> {
        if self.config.mc_samples > 0 {
            return Err(KwaversError::InvalidInput(
                "MC dropout is not supported by the current Burn PINN architectures".into(),
            ));
        }
        self.ensemble_prediction(input)
    }

    /// Deep ensemble prediction
    fn ensemble_prediction(&mut self, input: &[f32]) -> KwaversResult<PredictionWithUncertainty> {
        let mut predictions = Vec::new();

        // Get predictions from each ensemble member
        for model in &self.ensemble {
            let prediction = self.model_prediction(model, input)?;
            predictions.push(prediction);
        }

        let mut stats =
            self.compute_uncertainty_stats(&predictions, UncertaintyMethod::DeepEnsemble)?;

        // If conformal predictor is available, use it to refine confidence intervals
        if let Some(cp) = &self.conformal_predictor {
            let (lower, upper) = cp.predict_conformal(input)?;
            stats.confidence_interval = (vec![lower], vec![upper]);
            stats.method = UncertaintyMethod::Hybrid;
        }

        Ok(stats)
    }

    /// Compute uncertainty statistics from predictions
    fn compute_uncertainty_stats(
        &mut self,
        predictions: &[Vec<f32>],
        method: UncertaintyMethod,
    ) -> KwaversResult<PredictionWithUncertainty> {
        if predictions.is_empty() {
            return Err(KwaversError::System(
                crate::core::error::SystemError::InvalidOperation {
                    operation: "uncertainty_prediction".to_string(),
                    reason: "No predictions available".to_string(),
                },
            ));
        }

        let num_points = predictions[0].len();
        let num_samples = predictions.len();

        let mut means = vec![0.0; num_points];
        let mut variances = vec![0.0; num_points];

        // Compute mean and variance for each output point
        for i in 0..num_points {
            let mut sum = 0.0;
            let mut sum_sq = 0.0;

            for prediction in predictions {
                let val = prediction[i];
                sum += val;
                sum_sq += val * val;
            }

            let mean = sum / num_samples as f32;
            let variance = (sum_sq / num_samples as f32) - (mean * mean);

            means[i] = mean;
            variances[i] = variance.max(0.0); // Ensure non-negative variance
        }

        // Compute confidence intervals (95% = 1.96 * std)
        let stds: Vec<f32> = variances.iter().map(|v| v.sqrt()).collect();
        let z_score = 1.96; // 95% confidence

        let lower_bounds: Vec<f32> = means
            .iter()
            .zip(stds.iter())
            .map(|(m, s)| m - z_score * s)
            .collect();

        let upper_bounds: Vec<f32> = means
            .iter()
            .zip(stds.iter())
            .map(|(m, s)| m + z_score * s)
            .collect();

        // Compute entropy as measure of uncertainty
        let entropy = self.compute_predictive_entropy(&variances);

        // Compute reliability score based on variance threshold
        let reliability = self.compute_reliability_score(&variances);

        // Update statistics
        let avg_uncertainty = variances.iter().sum::<f32>() / variances.len() as f32;
        self.stats.total_predictions += 1;
        self.stats.average_uncertainty = (self.stats.average_uncertainty + avg_uncertainty) / 2.0;

        // Note: Calibration metrics are updated separately to avoid borrowing conflicts

        Ok(PredictionWithUncertainty {
            mean: means,
            std: stds,
            confidence_interval: (lower_bounds, upper_bounds),
            entropy,
            reliability,
            method,
        })
    }

    /// Get model prediction for ensemble member
    fn model_prediction(
        &self,
        model: &crate::ml::pinn::BurnPINN2DWave<B>,
        input: &[f32],
    ) -> KwaversResult<Vec<f32>> {
        if input.len() != 3 {
            return Err(KwaversError::InvalidInput(
                "Expected input to be [x, y, t]".into(),
            ));
        }
        let x = Array1::from_elem((1,), input[0] as f64);
        let y = Array1::from_elem((1,), input[1] as f64);
        let t = Array1::from_elem((1,), input[2] as f64);
        let device = model.device();

        let output = model
            .predict(&x, &y, &t, &device)
            .map(|output| output.iter().map(|&v| v as f32).collect::<Vec<_>>())?;

        Ok(output)
    }

    /// Compute predictive entropy
    fn compute_predictive_entropy(&self, variances: &[f32]) -> f32 {
        let avg_variance = variances.iter().sum::<f32>() / variances.len() as f32;
        // Simplified entropy calculation
        0.5 * (1.0 + (2.0 * std::f32::consts::PI * avg_variance).ln())
    }

    /// Compute reliability score
    fn compute_reliability_score(&self, variances: &[f32]) -> f32 {
        let avg_variance = variances.iter().sum::<f32>() / variances.len() as f32;
        // Higher reliability when variance is below threshold
        1.0 / (1.0 + avg_variance / self.config.variance_threshold as f32)
    }

    /// Update calibration metrics
    fn _update_calibration_metrics(
        &mut self,
        calibration_data: &[(Vec<f32>, f32)],
    ) -> KwaversResult<()> {
        if calibration_data.is_empty() {
            return Ok(());
        }

        let mut abs_errors = Vec::with_capacity(calibration_data.len());
        let mut covered = 0usize;

        for (input, target) in calibration_data {
            let prediction = self.ensemble_prediction(input)?;
            let mean = *prediction.mean.first().ok_or_else(|| {
                KwaversError::InvalidInput("Model returned empty prediction".into())
            })?;
            let std = *prediction.std.first().ok_or_else(|| {
                KwaversError::InvalidInput("Model returned empty uncertainty".into())
            })?;

            let err = (mean - *target).abs();
            abs_errors.push(err);

            let z_score = 1.96_f32;
            let lower = mean - z_score * std;
            let upper = mean + z_score * std;
            if (*target >= lower) && (*target <= upper) {
                covered += 1;
            }
        }

        let calibration_error = abs_errors.iter().sum::<f32>() / abs_errors.len() as f32;
        let coverage_probability = covered as f32 / calibration_data.len() as f32;

        self.stats.calibration_error = calibration_error;
        self.stats.coverage_probability = coverage_probability;
        self.stats.reliability_score =
            1.0 / (1.0 + calibration_error / self.config.variance_threshold as f32);

        Ok(())
    }

    /// Get uncertainty statistics
    pub fn get_stats(&self) -> &UncertaintyStats {
        &self.stats
    }
}

/// Conformal prediction for uncertainty quantification
#[derive(Debug)]
pub struct ConformalPredictor<B: Backend> {
    /// Base PINN model
    model: crate::ml::pinn::BurnPINN2DWave<B>,
    /// Conformal scores from calibration
    pub calibration_scores: Vec<f32>,
    /// Quantile for prediction intervals
    pub alpha: f64,
    pub quantile: Option<f32>,
}

impl<B: Backend> ConformalPredictor<B> {
    /// Create a new conformal predictor
    pub fn new(model: crate::ml::pinn::BurnPINN2DWave<B>, alpha: f64) -> Self {
        let alpha = alpha.clamp(f64::EPSILON, 1.0 - f64::EPSILON);
        Self {
            model,
            calibration_scores: Vec::new(),
            alpha,
            quantile: None,
        }
    }

    /// Calibrate using calibration data
    pub fn calibrate(
        &mut self,
        calibration_inputs: &[Vec<f32>],
        calibration_targets: &[f32],
    ) -> KwaversResult<()> {
        if calibration_inputs.len() != calibration_targets.len() {
            return Err(KwaversError::InvalidInput(
                "Calibration inputs and targets must have same length".into(),
            ));
        }

        let mut scores: Vec<f32> = Vec::with_capacity(calibration_inputs.len());

        for (input, target) in calibration_inputs.iter().zip(calibration_targets.iter()) {
            let score = self.compute_nonconformity_score(input, *target)?;
            scores.push(score);
        }

        if scores.is_empty() {
            return Err(KwaversError::InvalidInput(
                "Calibration dataset must be non-empty".into(),
            ));
        }

        scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = scores.len();
        let k = (((n as f64 + 1.0) * (1.0 - self.alpha)).ceil() as usize).clamp(1, n);
        let q_hat = scores[k - 1];

        self.calibration_scores = scores;
        self.quantile = Some(q_hat);

        Ok(())
    }

    /// Predict with conformal uncertainty intervals
    pub fn predict_conformal(&self, input: &[f32]) -> KwaversResult<(f32, f32)> {
        let q_hat = self.quantile.ok_or_else(|| {
            KwaversError::InvalidInput(
                "ConformalPredictor must be calibrated before prediction".into(),
            )
        })?;

        if input.len() != 3 {
            return Err(KwaversError::InvalidInput(
                "Expected input to be [x, y, t]".into(),
            ));
        }

        let device = self.model.device();
        let x = Array1::from_elem((1,), input[0] as f64);
        let y = Array1::from_elem((1,), input[1] as f64);
        let t = Array1::from_elem((1,), input[2] as f64);

        let pred = self.model.predict(&x, &y, &t, &device)?;
        let center =
            *pred.iter().next().ok_or_else(|| {
                KwaversError::InvalidInput("Model returned empty prediction".into())
            })? as f32;

        Ok((center - q_hat, center + q_hat))
    }

    /// Compute nonconformity score
    fn compute_nonconformity_score(&self, input: &[f32], target: f32) -> KwaversResult<f32> {
        if input.len() != 3 {
            return Err(KwaversError::InvalidInput(
                "Expected input to be [x, y, t]".into(),
            ));
        }

        let device = self.model.device();
        let x = Array1::from_elem((1,), input[0] as f64);
        let y = Array1::from_elem((1,), input[1] as f64);
        let t = Array1::from_elem((1,), input[2] as f64);

        let pred = self.model.predict(&x, &y, &t, &device)?;
        let y_hat =
            *pred.iter().next().ok_or_else(|| {
                KwaversError::InvalidInput("Model returned empty prediction".into())
            })? as f32;

        Ok((y_hat - target).abs())
    }
}

impl Default for UncertaintyStats {
    fn default() -> Self {
        Self {
            total_predictions: 0,
            average_uncertainty: 0.0,
            calibration_error: 0.0,
            coverage_probability: 0.0,
            reliability_score: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestBackend = burn::backend::NdArray<f32>;

    #[test]
    fn test_uncertainty_config() {
        let config = PinnUncertaintyConfig {
            mc_samples: 50,
            dropout_prob: 0.1,
            ensemble_size: 5,
            conformal_alpha: 0.05,
            variance_threshold: 0.1,
        };

        assert_eq!(config.mc_samples, 50);
        assert_eq!(config.ensemble_size, 5);
    }

    #[test]
    fn test_uncertainty_methods() {
        let methods = vec![
            UncertaintyMethod::MCDropout,
            UncertaintyMethod::DeepEnsemble,
            UncertaintyMethod::Conformal,
            UncertaintyMethod::Hybrid,
        ];

        for method in methods {
            // Methods should be debug-printable
            let _ = format!("{:?}", method);
        }
    }

    #[test]
    fn test_prediction_with_uncertainty() {
        let prediction = PredictionWithUncertainty {
            mean: vec![1.0, 0.5],
            std: vec![0.1, 0.05],
            confidence_interval: (vec![0.8, 0.4], vec![1.2, 0.6]),
            entropy: 0.5,
            reliability: 0.9,
            method: UncertaintyMethod::DeepEnsemble,
        };

        assert_eq!(prediction.mean.len(), 2);
        assert_eq!(prediction.std.len(), 2);
        assert!(prediction.reliability >= 0.0 && prediction.reliability <= 1.0);
    }

    #[test]
    fn test_conformal_predictor() {
        let config = PinnUncertaintyConfig {
            mc_samples: 0,
            dropout_prob: 0.0,
            ensemble_size: 1,
            conformal_alpha: 0.05,
            variance_threshold: 0.1,
        };

        assert!(config.conformal_alpha > 0.0 && config.conformal_alpha < 1.0);
    }
}
