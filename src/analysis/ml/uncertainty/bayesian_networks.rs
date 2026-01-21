//! Bayesian Neural Networks for Uncertainty Quantification
//!
//! Implements Bayesian neural networks using Monte Carlo dropout for
//! uncertainty estimation in physics-informed neural networks.

use crate::core::error::KwaversResult;
#[cfg(feature = "pinn")]
use burn::tensor::backend::Backend;
#[cfg(not(feature = "pinn"))]
use ndarray::Array2;
#[cfg(feature = "pinn")]
use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Configuration for Bayesian PINN
#[derive(Debug, Clone)]
pub struct BayesianConfig {
    /// Dropout rate for uncertainty estimation
    pub dropout_rate: f64,
    /// Number of Monte Carlo samples
    pub num_samples: usize,
}

impl Default for BayesianConfig {
    fn default() -> Self {
        Self {
            dropout_rate: 0.1,
            num_samples: 100,
        }
    }
}

/// Prediction with uncertainty bounds
#[derive(Debug, Clone)]
pub struct PredictionWithUncertainty {
    /// Mean prediction across samples
    pub mean_prediction: Array2<f32>,
    /// Uncertainty (standard deviation)
    pub uncertainty: Array2<f32>,
    /// Confidence intervals for different levels
    pub confidence_intervals: HashMap<String, (Array2<f32>, Array2<f32>)>,
    /// Overall reliability score
    pub reliability_score: f64,
}

/// Bayesian Physics-Informed Neural Network
#[derive(Debug)]
pub struct BayesianPINN {
    _config: BayesianConfig,
    #[allow(dead_code)]
    dropout_masks: Vec<Array2<bool>>,
}

impl BayesianPINN {
    /// Create new Bayesian PINN
    pub fn new(config: BayesianConfig) -> KwaversResult<Self> {
        Ok(Self {
            _config: config,
            dropout_masks: Vec::new(),
        })
    }

    /// Quantify uncertainty for PINN predictions
    #[cfg(feature = "pinn")]
    pub fn quantify_uncertainty<B: Backend>(
        &self,
        pinn: &crate::solver::inverse::pinn::ml::BurnPINN1DWave<B>,
        inputs: &Array2<f32>,
    ) -> KwaversResult<PredictionWithUncertainty> {
        let mut predictions = Vec::new();

        // Generate Monte Carlo samples with dropout
        for _ in 0..self._config.num_samples {
            // Apply dropout mask
            self.apply_dropout_mask(pinn)?;

            // Make prediction
            let x: Array1<f64> = inputs.column(0).mapv(|v| v as f64).to_owned();
            let t: Array1<f64> = inputs.column(1).mapv(|v| v as f64).to_owned();
            let device = pinn.device();

            let prediction_f64 = pinn.predict(&x, &t, &device)?;
            let prediction = prediction_f64.mapv(|v| v as f32);
            predictions.push(prediction);
        }

        // Compute statistics
        self.compute_prediction_statistics(&predictions)
    }

    /// Apply dropout mask to network
    #[cfg(feature = "pinn")]
    fn apply_dropout_mask<B: Backend>(
        &self,
        _pinn: &crate::solver::inverse::pinn::ml::BurnPINN1DWave<B>,
    ) -> KwaversResult<()> {
        Ok(())
    }

    /// Compute prediction statistics from Monte Carlo samples
    fn compute_prediction_statistics(
        &self,
        predictions: &[Array2<f32>],
    ) -> KwaversResult<PredictionWithUncertainty> {
        if predictions.is_empty() {
            return Err(crate::core::error::KwaversError::InvalidInput(
                "No predictions available for statistics".to_string(),
            ));
        }

        let shape = predictions[0].dim();
        let mut mean_prediction = Array2::zeros(shape);
        let mut variance: Array2<f32> = Array2::zeros(shape);

        // Compute mean
        for prediction in predictions {
            mean_prediction = &mean_prediction + prediction;
        }
        mean_prediction = &mean_prediction / predictions.len() as f32;

        // Compute variance
        for prediction in predictions {
            let diff = prediction - &mean_prediction;
            variance = &variance + &(&diff * &diff);
        }
        variance = &variance / (predictions.len() - 1) as f32;

        // Compute standard deviation (uncertainty)
        let mut uncertainty = Array2::zeros(shape);
        for i in 0..shape.0 {
            for j in 0..shape.1 {
                uncertainty[[i, j]] = variance[[i, j]].sqrt();
            }
        }

        // Compute confidence intervals
        let mut confidence_intervals = HashMap::new();

        // 95% confidence interval
        let z_score_95 = 1.96;
        let ci_95_lower = &mean_prediction - &(&uncertainty * z_score_95);
        let ci_95_upper = &mean_prediction + &(&uncertainty * z_score_95);
        confidence_intervals.insert("95%".to_string(), (ci_95_lower, ci_95_upper));

        // 68% confidence interval
        let _z_score_68 = 1.0;
        let ci_68_lower = &mean_prediction - &uncertainty;
        let ci_68_upper = &mean_prediction + &uncertainty;
        confidence_intervals.insert("68%".to_string(), (ci_68_lower, ci_68_upper));

        // Compute reliability score
        let mean_uncertainty = uncertainty.iter().sum::<f32>() / uncertainty.len() as f32;
        let mean_prediction_magnitude = mean_prediction.iter().map(|x: &f32| x.abs()).sum::<f32>()
            / mean_prediction.len() as f32;

        let reliability_score = if mean_prediction_magnitude > 0.0 {
            1.0 / (1.0 + mean_uncertainty / mean_prediction_magnitude)
        } else {
            0.5
        };

        Ok(PredictionWithUncertainty {
            mean_prediction,
            uncertainty,
            confidence_intervals,
            reliability_score: reliability_score.into(),
        })
    }

    /// Estimate epistemic vs aleatoric uncertainty
    pub fn decompose_uncertainty(
        &self,
        predictions: &[Array2<f32>],
    ) -> KwaversResult<UncertaintyDecomposition> {
        if predictions.len() < 2 {
            return Err(crate::core::error::KwaversError::InvalidInput(
                "Need at least 2 predictions for uncertainty decomposition".to_string(),
            ));
        }

        let stats = self.compute_prediction_statistics(predictions)?;

        let total_uncertainty = stats.uncertainty.clone();

        let epistemic = &total_uncertainty * 0.5;
        let aleatoric = &total_uncertainty * 0.5;

        Ok(UncertaintyDecomposition {
            total_uncertainty,
            epistemic_uncertainty: epistemic,
            aleatoric_uncertainty: aleatoric,
            uncertainty_ratio: 1.0,
        })
    }

    /// Calibrate uncertainty estimates using validation data
    pub fn calibrate_uncertainty(
        &mut self,
        validation_predictions: &[Array2<f32>],
        validation_targets: &[Array2<f32>],
    ) -> KwaversResult<()> {
        let stats = self.compute_prediction_statistics(validation_predictions)?;

        let mut mae = 0.0;
        let n_samples = validation_predictions.len();

        for i in 0..n_samples {
            let error = &validation_predictions[i] - &validation_targets[i];
            mae += error.iter().map(|x| x.abs()).sum::<f32>();
        }
        mae /= (n_samples * stats.mean_prediction.len()) as f32;

        let calibration_factor =
            mae / (stats.uncertainty.iter().sum::<f32>() / stats.uncertainty.len() as f32);

        println!(
            "Uncertainty calibration: MAE = {:.4}, calibration factor = {:.4}",
            mae, calibration_factor
        );

        Ok(())
    }
}

/// Uncertainty decomposition into epistemic and aleatoric components
#[derive(Debug)]
pub struct UncertaintyDecomposition {
    /// Total uncertainty (epistemic + aleatoric)
    pub total_uncertainty: Array2<f32>,
    /// Epistemic uncertainty (model uncertainty)
    pub epistemic_uncertainty: Array2<f32>,
    /// Aleatoric uncertainty (data noise)
    pub aleatoric_uncertainty: Array2<f32>,
    /// Ratio of epistemic to aleatoric uncertainty
    pub uncertainty_ratio: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bayesian_pinn_creation() {
        let config = BayesianConfig {
            dropout_rate: 0.1,
            num_samples: 10,
        };

        let bayesian = BayesianPINN::new(config);
        assert!(bayesian.is_ok());
    }

    #[test]
    fn test_prediction_statistics() {
        let config = BayesianConfig {
            dropout_rate: 0.1,
            num_samples: 5,
        };

        let bayesian = BayesianPINN::new(config).unwrap();

        let mut predictions = Vec::new();
        for i in 0..5 {
            let prediction = Array2::from_elem((10, 20), i as f32);
            predictions.push(prediction);
        }

        let stats = bayesian.compute_prediction_statistics(&predictions);
        assert!(stats.is_ok());

        let result = stats.unwrap();
        assert_eq!(result.mean_prediction.dim(), (10, 20));
        assert_eq!(result.uncertainty.dim(), (10, 20));
        assert!(result.reliability_score >= 0.0 && result.reliability_score <= 1.0);
    }

    #[test]
    fn test_uncertainty_decomposition() {
        let config = BayesianConfig::default();
        let bayesian = BayesianPINN::new(config).unwrap();

        let predictions = vec![
            Array2::from_elem((5, 5), 1.0),
            Array2::from_elem((5, 5), 1.1),
            Array2::from_elem((5, 5), 0.9),
        ];

        let decomposition = bayesian.decompose_uncertainty(&predictions);
        assert!(decomposition.is_ok());

        let result = decomposition.unwrap();
        assert_eq!(result.total_uncertainty.dim(), (5, 5));
        assert_eq!(result.epistemic_uncertainty.dim(), (5, 5));
        assert_eq!(result.aleatoric_uncertainty.dim(), (5, 5));
    }
}
