//! Ensemble Methods for Uncertainty Quantification
//!
//! Implements ensemble-based uncertainty estimation using bagging and
//! bootstrap aggregation for robust uncertainty bounds.

mod model;
#[cfg(test)]
mod tests;

use crate::core::error::KwaversResult;
#[cfg(feature = "pinn")]
use burn::tensor::backend::Backend;
#[cfg(not(feature = "pinn"))]
use ndarray::Array2;
#[cfg(feature = "pinn")]
use ndarray::Array2;
use model::EnsembleModel;
use std::collections::HashMap;

/// Configuration for ensemble methods
#[derive(Debug, Clone)]
pub struct EnsembleConfig {
    /// Number of models in ensemble
    pub ensemble_size: usize,
    /// Number of samples per model
    pub num_samples: usize,
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        Self {
            ensemble_size: 10,
            num_samples: 100,
        }
    }
}

/// Ensemble result with uncertainty
#[derive(Debug)]
pub struct EnsembleResult {
    /// Ensemble predictions
    pub predictions: Vec<Array2<f32>>,
    /// Mean prediction across ensemble
    pub mean_prediction: Array2<f32>,
    /// Prediction variance (uncertainty)
    pub prediction_variance: Array2<f32>,
    /// Individual model weights
    pub model_weights: Vec<f64>,
    /// Ensemble diversity measure
    pub diversity_score: f64,
}

/// Ensemble quantifier for uncertainty estimation
#[derive(Debug)]
pub struct EnsembleQuantifier {
    ensemble_models: Vec<EnsembleModel>,
}

impl EnsembleQuantifier {
    /// Create new ensemble quantifier
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new(config: EnsembleConfig) -> KwaversResult<Self> {
        let mut ensemble_models = Vec::new();
        for i in 0..config.ensemble_size {
            ensemble_models.push(EnsembleModel::new(i as u64));
        }
        Ok(Self { ensemble_models })
    }

    /// Quantify uncertainty using ensemble methods
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    #[cfg(feature = "pinn")]
    pub fn quantify_uncertainty<B: Backend>(
        &self,
        pinn: &crate::solver::inverse::pinn::ml::BurnPINN1DWave<B>,
        inputs: &Array2<f32>,
    ) -> KwaversResult<super::PredictionWithUncertainty> {
        let mut predictions = Vec::new();
        let mut weights = Vec::new();

        for model in &self.ensemble_models {
            let prediction = model.predict_with_noise(pinn, inputs)?;
            predictions.push(prediction);
            weights.push(model.weight);
        }

        self.compute_ensemble_statistics(&predictions, &weights)
    }

    /// Train ensemble models with bootstrap sampling
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn train_ensemble(
        &mut self,
        training_data: &[Array2<f32>],
        training_targets: &[Array2<f32>],
    ) -> KwaversResult<()> {
        let bootstrap_samples: Vec<Vec<usize>> = (0..self.ensemble_models.len())
            .map(|_| self.bootstrap_sample(training_data.len()))
            .collect();

        for (model_idx, bootstrap_indices) in bootstrap_samples.into_iter().enumerate() {
            let bootstrap_data: Vec<_> = bootstrap_indices
                .iter()
                .map(|&idx| training_data[idx].clone())
                .collect();
            let bootstrap_targets: Vec<_> = bootstrap_indices
                .iter()
                .map(|&idx| training_targets[idx].clone())
                .collect();

            self.ensemble_models[model_idx].train(&bootstrap_data, &bootstrap_targets)?;
        }

        Ok(())
    }

    /// Generate bootstrap sample indices
    pub(super) fn bootstrap_sample(&self, n_samples: usize) -> Vec<usize> {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let mut rng = StdRng::from_entropy();
        let mut indices = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            indices.push(rng.gen_range(0..n_samples));
        }
        indices
    }

    /// Compute ensemble statistics
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn compute_ensemble_statistics(
        &self,
        predictions: &[Array2<f32>],
        weights: &[f64],
    ) -> KwaversResult<super::PredictionWithUncertainty> {
        if predictions.is_empty() {
            return Err(crate::core::error::KwaversError::InvalidInput(
                "No predictions available".to_owned(),
            ));
        }

        let shape = predictions[0].dim();
        let mut weighted_sum = Array2::zeros(shape);
        let mut total_weight = 0.0;

        for (prediction, &weight) in predictions.iter().zip(weights.iter()) {
            weighted_sum = &weighted_sum + &(&prediction.view() * weight as f32);
            total_weight += weight;
        }

        let mean_prediction = if total_weight > 0.0 {
            &weighted_sum / total_weight as f32
        } else {
            predictions[0].clone()
        };

        let mut variance = Array2::zeros(shape);
        for (prediction, &weight) in predictions.iter().zip(weights.iter()) {
            let diff = prediction - &mean_prediction;
            let weighted_diff = &diff * &diff * weight as f32;
            variance = &variance + &weighted_diff;
        }

        if total_weight > 1.0 {
            variance = &variance / total_weight as f32;
        }

        let uncertainty = variance.mapv(|x: f32| x.sqrt());

        let mut confidence_intervals = HashMap::new();
        let ci_factor = 1.96;
        let ci_95_lower = &mean_prediction - &(&uncertainty * ci_factor);
        let ci_95_upper = &mean_prediction + &(&uncertainty * ci_factor);
        confidence_intervals.insert("95%".to_owned(), (ci_95_lower, ci_95_upper));

        let diversity = self.compute_ensemble_diversity(predictions);
        let reliability_score = 1.0 / (1.0 + diversity);

        Ok(super::PredictionWithUncertainty {
            mean_prediction,
            uncertainty,
            confidence_intervals,
            reliability_score,
        })
    }

    /// Compute ensemble diversity (disagreement between models)
    pub(super) fn compute_ensemble_diversity(&self, predictions: &[Array2<f32>]) -> f64 {
        if predictions.len() < 2 {
            return 0.0;
        }

        let mut total_diversity = 0.0;
        let mut count = 0;

        for i in 0..predictions.len() {
            for j in (i + 1)..predictions.len() {
                let diff = &predictions[i] - &predictions[j];
                let diversity = diff.iter().map(|x| x * x).sum::<f32>().sqrt() as f64;
                total_diversity += diversity;
                count += 1;
            }
        }

        if count > 0 { total_diversity / count as f64 } else { 0.0 }
    }

    /// Get ensemble result with detailed statistics
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn get_ensemble_result(
        &self,
        predictions: &[Array2<f32>],
        weights: &[f64],
    ) -> KwaversResult<EnsembleResult> {
        let stats = self.compute_ensemble_statistics(predictions, weights)?;
        let diversity_score = self.compute_ensemble_diversity(predictions);

        Ok(EnsembleResult {
            predictions: predictions.to_vec(),
            mean_prediction: stats.mean_prediction,
            prediction_variance: &stats.uncertainty * &stats.uncertainty,
            model_weights: weights.to_vec(),
            diversity_score,
        })
    }

    /// Update ensemble weights based on performance
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn update_weights(&mut self, performance_scores: &[f64]) -> KwaversResult<()> {
        if performance_scores.len() != self.ensemble_models.len() {
            return Err(crate::core::error::KwaversError::InvalidInput(
                "Performance scores must match ensemble size".to_owned(),
            ));
        }

        let total_performance: f64 = performance_scores.iter().sum();
        if total_performance > 0.0 {
            for (model, &performance) in self
                .ensemble_models
                .iter_mut()
                .zip(performance_scores.iter())
            {
                model.weight = performance / total_performance;
            }
        }

        Ok(())
    }
}
